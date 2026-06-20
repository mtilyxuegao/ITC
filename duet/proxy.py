"""DUET-Lite proxy — one page, duplex voice + hermes thinking layer together.

Sits between the browser and the MiniCPM duplex gateway:
  - HTTP: reverse-proxies the official duplex UI unchanged, but injects a
    "Thinking Layer" overlay panel into the HTML.
  - WS /v1/realtime: bidirectional passthrough to MiniCPM (voice works as-is)
    while TAPPING the assistant's text stream -> intent -> [THINK].
  - WS /duet/events: pushes the Conductor's control-token timeline + hermes
    milestones + results to the overlay, live.

So the browser opens ONE URL and gets full-duplex voice (MiniCPM) and the
decoupled thinking layer (hermes -> Qwen/gemma) on the same screen.

READ seam (no STT needed for v1): we classify the *assistant's* emitted text
each turn — when MiniCPM says something knowledge-seeking ("let me check the
flights…"), that fires the thinking layer. Swap in STT on the user audio later
for a user-side trigger.

Run (in hermes-env, which has aiohttp + hermes-agent + the duet pkg on PYTHONPATH):
  PROXY_PORT=8010 MINICPM_GATEWAY=http://liquid-gpu-053:8006 \
  QWEN_URL=http://liquid-gpu-001:8001 \
  PYTHONPATH=/home/justin/ITC:/home/justin/hermes-agent \
  /home/justin/hermes-env/bin/python -m duet.proxy
Then point cloudflared at http://localhost:8010 instead of the gateway.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os

import aiohttp
from aiohttp import web

from .conductor import Conductor
from .events import Event, EventLog, Kind
from .hermes_thinking import HermesThinkingClient
from .state import Phase
from .tts import tts_pcm_f32
from .write_seam import RecordingSpeaker


def _condense(text: str, limit: int = 220) -> str:
    """Trim a long RESULT to one spoken sentence for the voice write-back."""
    text = " ".join((text or "").replace("*", "").replace("#", "").split())
    for sep in (". ", "。", "; "):
        i = text.find(sep)
        if 0 < i < limit:
            return text[:i + 1]
    return text[:limit]

# Appended to the duplex model's system prompt so it DEFERS instead of hallucinating
# ("model抢答"): the frozen model can't be trained to emit [THINK], but it largely
# follows a system instruction to stop and let the backend answer.
DEFER_PROMPT = (
    " Rule: for any factual or real-time question (prices, data, news, facts), do NOT state "
    "any number or fact yourself — just say a brief \"Sure, let me check.\" and wait. When you "
    "later hear a note starting with \"Answer:\", say that answer in one short sentence. "
    "Chat normally for small talk."
)

# The model emitting one of these phrases IS its [THINK] signal — that's when (and only
# when) we dispatch the thinking layer. Keeps casual chit-chat from triggering the backend.
_DEFER_SIGNALS = (
    "let me check", "let me look", "i'll check", "i'll look", "i will check",
    "look that up", "look it up", "let me find", "check that for you", "checking that",
    "我查", "查一下", "让我查", "我来查", "帮你查", "查询一下",
)

GATEWAY = os.environ.get("MINICPM_GATEWAY", "http://liquid-gpu-053:8006").rstrip("/")
# Thinking-layer (hermes research) model endpoint. THINK_URL is the standard name;
# QWEN_URL is still accepted as a legacy fallback.
THINK_URL = os.environ.get("THINK_URL", os.environ.get("QWEN_URL", "http://liquid-gpu-026:8001")).rstrip("/")
ASR_URL = os.environ.get("ASR_URL", "http://liquid-gpu-060:8020").rstrip("/")
MODEL = os.environ.get("THINKER_MODEL", "qwen")
# Aux model (router + spokenify) — runs on gemma so the research model's GPUs stay free.
AUX_URL = os.environ.get("AUX_URL", THINK_URL).rstrip("/")
AUX_MODEL = os.environ.get("AUX_MODEL", MODEL)
PORT = int(os.environ.get("PROXY_PORT", "8010"))
# Grace before the anti-抢答 mute kicks in, so the model's brief ack ("Sure, let me check")
# finishes instead of being clipped mid-word. Short enough that a hallucinated FACT — which the
# model only reaches AFTER the ack — still gets cut. Tune via env if 抢答 leaks (lower) or the
# ack still clips (raise).
GATE_DELAY = float(os.environ.get("GATE_DELAY", "1.0"))
TOOLSETS = [t for t in os.environ.get("HERMES_TOOLSETS", "").split(",") if t]
MAXMSG = 128 * 1024 * 1024  # match MiniCPM gateway's bumped WS payload limit

_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
        "content-length"}

OVERLAY_TAG = b'<script src="/duet/overlay.js"></script>\n</body>'


def _event_json(ev: Event) -> str:
    return json.dumps({"kind": ev.kind, "epoch": ev.epoch, "label": ev.label(),
                       "text": ev.text, "meta": ev.meta}, ensure_ascii=False)


class Hub:
    """Fan-out of Conductor events to all connected overlay WebSockets."""

    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue] = set()
        self.active_up = None          # the live upstream WS to MiniCPM (for write-back)
        self.active_gate = None        # per-connection state dict (drives force_listen)
        self._gate_token = 0           # invalidates stale force_listen failsafe timers
        self._t_dispatch = 0.0         # loop time at dispatch, for answer-latency timing
        self.recent: list[str] = []    # recent STT transcripts (context for the router)
        self.log = EventLog()
        self.log.subscribe(self._on_event)
        self.conductor = Conductor(
            thinking=HermesThinkingClient(THINK_URL, model=MODEL, enabled_toolsets=TOOLSETS,
                                          max_iterations=3),   # speed > depth: 1 search + answer
            speaker=RecordingSpeaker(), log=self.log, scene="duplex")

    def _on_event(self, ev: Event) -> None:
        data = _event_json(ev)
        for q in list(self.subscribers):
            try:
                q.put_nowait(data)
            except Exception:
                pass
        if ev.kind == Kind.RESULT and ev.text:      # write the answer back into the model's voice
            asyncio.create_task(self._voice_back(ev.text))

    def reset(self) -> None:
        """Clear backend conversation history (fresh slate per session)."""
        self.conductor.reset()
        self.log.clear()
        self.recent.clear()
        self._set_listen_gate(False)   # drop any stale mute + invalidate failsafe timers
        msg = json.dumps({"kind": "__reset__"})
        for q in list(self.subscribers):
            try:
                q.put_nowait(msg)
            except Exception:
                pass
        print("[duet] conversation history reset", flush=True)

    def _set_listen_gate(self, on: bool) -> None:
        """Hard-mute (force_listen) the FROZEN duplex model during the research window so
        it can't 抢答 (speak its own hallucinated fact); release it when the answer write-back
        begins. force_listen is a per-frame flag the model re-reads every chunk and, mid-turn,
        feeds <|turn_eos|> to stop any in-progress speech within ~one chunk. Driven by adding
        the flag to the browser's input.append frames in rewrite_client.
        A failsafe auto-clears it after 25s so a failed/timed-out research can't mute forever."""
        self._gate_token += 1
        if self.active_gate is not None:
            self.active_gate["force_listen"] = on
        if on:
            tok = self._gate_token
            async def _failsafe():
                await asyncio.sleep(25)
                if self._gate_token == tok and self.active_gate is not None:
                    self.active_gate["force_listen"] = False
                    print("[duet] force_listen failsafe-cleared (research timeout)", flush=True)
            asyncio.create_task(_failsafe())

    async def _spokenify(self, text: str) -> str:
        """Rewrite the verbose RESULT into one short, TTS/ASR-robust spoken sentence.
        The write-back goes TTS -> MiniCPM ASR -> voice, which mangles '$210.33'-style
        numbers; plain rounded words survive the round-trip."""
        payload = {"model": AUX_MODEL, "stream": False, "max_tokens": 64, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [
                       {"role": "system", "content":
                        "Rewrite the answer as ONE short sentence a voice assistant says "
                        "aloud. Spell numbers as plain rounded words (e.g. 'about two "
                        "hundred ten dollars'). No symbols, no $, no decimals, no ranges, "
                        "no URLs. Under 18 words."},
                       {"role": "user", "content": text[:600]}]}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(AUX_URL + "/v1/chat/completions", json=payload,
                                  timeout=aiohttp.ClientTimeout(total=20)) as r:
                    d = await r.json()
            out = (d["choices"][0]["message"]["content"] or "").strip()
            return out or _condense(text)
        except Exception as e:
            print(f"[duet] spokenify error: {e}", flush=True)
            return _condense(text)

    async def _voice_back(self, text: str) -> None:
        """Make the duplex model speak the answer VERBATIM via a force_speak text
        injection (teacher-forced into the decoder, then TTS-rendered). Replaces the
        old TTS -> MiniCPM re-ASR -> re-voice path, which garbled names/numbers
        ("Ramin Hasani" -> "Robin Ha sa"). We still run _spokenify to condense the
        verbose RESULT into ONE short sentence (and round numbers for natural voice),
        but it no longer needs to be ASR-robust. control.force_speak is sent DIRECTLY
        to the gateway WS (active_up), NOT through rewrite_client. We must clear the
        force_listen gate FIRST, else the next browser input.append frame re-mutes the
        model mid-utterance and cuts off the forced speech."""
        up = self.active_up
        if up is None or getattr(up, "closed", True):
            return
        spoken = await self._spokenify(text)
        if not spoken:
            spoken = _condense(text)
        if not spoken:
            return
        self._set_listen_gate(False)                 # release the model BEFORE forcing speech
        try:
            await up.send_str(json.dumps(
                {"type": "control.force_speak", "payload": {"text": spoken}},
                ensure_ascii=False))
            _ms = (asyncio.get_running_loop().time() - self._t_dispatch) * 1000 if self._t_dispatch else 0
            print(f"[duet] force_speak (research+spokenify {_ms:.0f}ms) -> {spoken[:70]!r}", flush=True)
        except Exception as e:
            # No fallback to the old TTS->input.append path: that lossy path is exactly
            # what this removes. Gate already cleared, so the model stays responsive.
            print(f"[duet] force_speak failed: {e}", flush=True)

    # called when the model finishes a turn: its (defer-prompted) text RESTATES the
    # user's question. We dispatch the thinking layer on that clean restatement.
    async def _route(self, transcript: str, context: list):
        """Context-aware query builder: resolve the LATEST turn (handling corrections
        like 'I mean the stock price' or 'just the nvidia one' using recent context, and
        fixing obvious STT errors) into ONE standalone web query, or NONE for chit-chat."""
        ctx = " | ".join(context[-5:]) if context else "(none)"
        payload = {"model": AUX_MODEL, "stream": False, "max_tokens": 64, "temperature": 0,
                   # Qwen3.5 is a reasoning model; without this it burns all tokens on
                   # <think> and returns content=null. Routing needs no reasoning.
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [
                       {"role": "system", "content":
                        "You build ONE web search query for a live voice assistant. Given the "
                        "recent user turns and the LATEST turn, turn the latest into a single "
                        "standalone search query. Rules: (1) handle corrections/clarifications "
                        "(e.g. 'I mean the stock price', 'just the nvidia one') using context; "
                        "(2) fix obvious speech-to-text errors (e.g. 'store'->'stock'); (3) if "
                        "the latest asks for MULTIPLE things (e.g. 'weather in SF and the nvidia "
                        "stock price'), keep them BOTH in one query — do NOT drop it. Output the "
                        "query for ANY question or information request, however casual. Output "
                        "exactly NONE ONLY when the latest is purely a greeting, thanks, "
                        "acknowledgement, or filler with literally no question in it."},
                       {"role": "user", "content": f"Recent turns: {ctx}\nLatest: {transcript}"}]}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(AUX_URL + "/v1/chat/completions", json=payload,
                                  timeout=aiohttp.ClientTimeout(total=20)) as r:
                    d = await r.json()
            out = (d["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            print(f"[duet] router error: {e}", flush=True)
            return None
        if not out or out.upper().lstrip(" .\"'").startswith("NONE"):
            return None
        return out

    async def _asr(self, pcm: bytes) -> str:
        """Transcribe the user's audio via the GPU whisper-large-v3 service."""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(ASR_URL + "/asr",
                                  json={"audio_b64": base64.b64encode(pcm).decode()},
                                  timeout=aiohttp.ClientTimeout(total=30)) as r:
                    d = await r.json()
            return (d.get("text") or "").strip()
        except Exception as e:
            print(f"[duet] ASR error: {e}", flush=True)
            return ""

    async def on_user_audio(self, pcm: bytes) -> None:
        """User's turn ended: STT -> noise filter -> semantic gate -> dispatch.
        STT is noisy (hallucinated text on silence/echo), so we (1) drop short
        fragments and (2) require the router to confirm a clear info request."""
        loop = asyncio.get_running_loop(); _t0 = loop.time()
        transcript = (await self._asr(pcm)).strip()
        _ms_asr = (loop.time() - _t0) * 1000
        if len(transcript.split()) < 3:          # filler / half-word / noise -> ignore
            self._set_listen_gate(False)          # not a real query -> release the optimistic mute
            if transcript:
                print(f"[duet] STT (asr {_ms_asr:.0f}ms, too short, skip) -> {transcript!r}", flush=True)
            return
        print(f"[duet] STT (asr {_ms_asr:.0f}ms) -> {transcript!r}", flush=True)
        self.recent.append(transcript)
        del self.recent[:-6]
        _t1 = loop.time()
        query = await self._route(transcript, self.recent)   # context-aware query / None
        _ms_route = (loop.time() - _t1) * 1000
        if not query:
            self._set_listen_gate(False)          # chit-chat -> release the optimistic mute
            print(f"[duet] route: drop (route {_ms_route:.0f}ms) -> {transcript[:50]!r}", flush=True)
            return
        self._t_dispatch = loop.time()            # for end-to-end answer-latency timing in _voice_back
        print(f"[duet] DISPATCH (route {_ms_route:.0f}ms) -> {query!r}", flush=True)
        self._set_listen_gate(True)               # mute the model during research (anti-抢答)
        if self.conductor.phase == Phase.THINKING:
            await self.conductor.on_interrupt()   # new turn supersedes -> [WAIT]
        await self.conductor.on_user_utterance(query)


async def proxy_http(request: web.Request) -> web.StreamResponse:
    hub: Hub = request.app["hub"]
    session: aiohttp.ClientSession = request.app["session"]
    target = GATEWAY + request.raw_path
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host",)}
    headers["Accept-Encoding"] = "identity"  # don't gzip; we may rewrite HTML
    body = await request.read()
    async with session.request(request.method, target, headers=headers,
                               data=body, allow_redirects=False) as r:
        raw = await r.read()
        ct = r.headers.get("Content-Type", "")
        if "text/html" in ct and b"</body>" in raw:
            raw = raw.replace(b"</body>", OVERLAY_TAG, 1)
        out_headers = {k: v for k, v in r.headers.items()
                       if k.lower() not in _HOP}
        return web.Response(status=r.status, body=raw, headers=out_headers)


async def proxy_ws(request: web.Request) -> web.WebSocketResponse:
    hub: Hub = request.app["hub"]
    session: aiohttp.ClientSession = request.app["session"]
    client = web.WebSocketResponse(max_msg_size=MAXMSG)
    await client.prepare(request)
    hub.reset()                  # fresh conversation history for this new session
    print(f"[duet] /v1/realtime opened mode={request.query.get('mode')}", flush=True)

    up_url = "ws" + GATEWAY[len("http"):] + "/v1/realtime"
    if request.query_string:
        up_url += "?" + request.query_string

    user_pcm = bytearray()       # USER mic audio this turn (16kHz mono f32), echo-free
    state = {"speaking": False, "force_listen": False}
    _seen: set = set()

    def rewrite_client(data):
        """Browser->gateway: inject DEFER prompt into session.init; capture USER audio
        ONLY while the model is listening (not speaking) so no model echo leaks in."""
        try:
            msg = json.loads(data)
        except Exception:
            return data
        t = msg.get("type")
        if t == "session.init":
            payload = msg.get("payload")
            if not isinstance(payload, dict):
                payload = {}
                msg["payload"] = payload
            field = "instructions" if "instructions" in payload else "system_prompt"
            payload[field] = (payload.get(field) or "") + DEFER_PROMPT
            print("[duet] injected DEFER prompt into session.init", flush=True)
            return json.dumps(msg, ensure_ascii=False)
        if t == "input.append":
            if not state["speaking"]:
                a = (msg.get("input") or {}).get("audio")
                if a:
                    try:
                        user_pcm.extend(base64.b64decode(a))
                    except Exception:
                        pass
            if state["force_listen"]:        # anti-抢答: force the model to keep listening
                msg.setdefault("input", {})["force_listen"] = True
                return json.dumps(msg, ensure_ascii=False)
        return data

    def tap_gateway(data):
        """Gateway->browser: when the model STARTS speaking, the user's turn ended ->
        STT the captured user audio (their real question)."""
        if isinstance(data, (bytes, bytearray)):
            return
        try:
            msg = json.loads(data)
        except Exception:
            return
        t = msg.get("type", ""); kind = msg.get("kind")
        key = (t, kind)
        if key not in _seen:
            _seen.add(key)
            print(f"[duet][type] {t}/{kind}", flush=True)
        is_speak = (t == "response.output_audio.delta"
                    or (t == "response.output.delta" and kind in ("text", "audio")))
        if is_speak:
            if not state["speaking"]:
                state["speaking"] = True
                state["spoke_text"] = ""              # per-turn buffer of the model's spoken words
                pcm = bytes(user_pcm); user_pcm.clear()
                if len(pcm) > int(16000 * 4 * 0.3):   # >~0.3s of f32 audio = a real user turn
                    # Arm the anti-抢答 mute after GATE_DELAY so the model's brief ack
                    # ("Sure, let me check") finishes instead of clipping mid-word, while still
                    # cutting a hallucinated FACT (which comes only AFTER the ack). STT+route
                    # (~2-3s) finishes after the arm, then on_user_audio keeps the mute for a
                    # research turn or releases it for chit-chat.
                    async def _arm_gate(st=state):
                        await asyncio.sleep(GATE_DELAY)
                        st["force_listen"] = True
                    asyncio.create_task(_arm_gate())
                    asyncio.create_task(hub.on_user_audio(pcm))
            # CONTENT-BASED anti-抢答 gate (additive to the GATE_DELAY timer): the kind=text
            # delta carries the spoken words BEFORE that chunk's audio. The instant a FACT token
            # (digit / $ % °) appears past the short ack, mute — cutting a hallucinated number/price
            # before it is heard. The ack ("Sure, let me check") has no digits so it is unaffected.
            if kind == "text" and not state["force_listen"]:
                state["spoke_text"] = (state.get("spoke_text") or "") + (msg.get("text") or "")
                buf = state["spoke_text"]
                if len(buf) > 8 and any(c in buf for c in "0123456789$%°"):
                    state["force_listen"] = True
                    print(f"[duet] content-gate: muted on fact-token in {buf[:48]!r}", flush=True)
        elif t == "response.output.delta" and kind == "listen":
            state["speaking"] = False

    try:
        async with session.ws_connect(up_url, max_msg_size=MAXMSG, heartbeat=30) as up:
            hub.active_up = up           # enable write-back into this duplex session
            hub.active_gate = state      # let the hub drive force_listen on this connection
            async def c2u():
                async for m in client:
                    if m.type == aiohttp.WSMsgType.TEXT:
                        await up.send_str(rewrite_client(m.data))
                    elif m.type == aiohttp.WSMsgType.BINARY:
                        await up.send_bytes(m.data)
                await up.close()

            async def u2c():
                async for m in up:
                    if m.type == aiohttp.WSMsgType.TEXT:
                        await client.send_str(m.data)
                        tap_gateway(m.data)
                    elif m.type == aiohttp.WSMsgType.BINARY:
                        await client.send_bytes(m.data)
                await client.close()

            await asyncio.gather(c2u(), u2c())
    except Exception:
        if not client.closed:
            await client.close()
    finally:
        hub.active_up = None
        hub.active_gate = None
    return client


async def events_ws(request: web.Request) -> web.WebSocketResponse:
    hub: Hub = request.app["hub"]
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    q: asyncio.Queue = asyncio.Queue()
    hub.subscribers.add(q)
    try:
        for ev in hub.log.events:          # replay backlog to a fresh panel
            await ws.send_str(_event_json(ev))
        while True:
            await ws.send_str(await q.get())
    except Exception:
        pass
    finally:
        hub.subscribers.discard(q)
    return ws


async def overlay_js(request: web.Request) -> web.Response:
    return web.Response(text=_OVERLAY_JS, content_type="application/javascript")


def make_app() -> web.Application:
    app = web.Application(client_max_size=MAXMSG)
    app["hub"] = Hub()

    async def _on_start(app):
        app["session"] = aiohttp.ClientSession()

    async def _on_clean(app):
        await app["session"].close()

    app.on_startup.append(_on_start)
    app.on_cleanup.append(_on_clean)

    app.router.add_get("/duet/overlay.js", overlay_js)
    app.router.add_get("/duet/events", events_ws)
    app.router.add_get("/v1/realtime", proxy_ws)
    app.router.add_route("*", "/{tail:.*}", proxy_http)  # everything else -> gateway
    return app


_OVERLAY_JS = r"""
(function(){
  if (window.__duet) return; window.__duet = true;
  var C = {'[THINK]':'#54E0C7','[WAIT]':'#FF6A5A','[CUT]':'#FF6A5A','^':'#F0A94B',
           'RESULT':'#54E0C7','inject→speak':'#9ec','dropped':'#667','user':'#F0A94B',
           'dispatch':'#789','speak':'#9ec'};
  var p = document.createElement('div');
  p.style.cssText = 'position:fixed;top:0;right:0;width:340px;height:100vh;z-index:99999;'+
    'background:#0C1116;color:#C9D4DA;font:12px ui-monospace,Menlo,monospace;'+
    'border-left:1px solid #243; box-shadow:-4px 0 24px rgba(0,0,0,.4);display:flex;flex-direction:column;';
  p.innerHTML = '<div style="padding:12px 14px;border-bottom:1px solid #233;letter-spacing:.12em;'+
    'text-transform:uppercase;color:#7E8C96;font-size:11px">Thinking Layer · Qwen3.5-35B-A3B / hermes'+
    '<span id="duet-dot" style="float:right;color:#FF6A5A">●</span></div>'+
    '<div id="duet-log" style="flex:1;overflow:auto;padding:8px 12px"></div>';
  document.body.appendChild(p);
  var log = p.querySelector('#duet-log'), dot = p.querySelector('#duet-dot');
  function add(ev){
    if(ev.kind==='__reset__'){ log.innerHTML=''; return; }
    var c = C[ev.kind]||'#8aa';
    var row = document.createElement('div');
    row.style.cssText='padding:5px 0;border-bottom:1px solid #1a222a;line-height:1.4';
    var dl = ev.kind==='user' ? 'QUERY' : ev.label;
    var lbl = '<span style="color:'+c+';font-weight:700">'+dl+'</span>';
    var body = ev.text? ' <span style="color:#aeb8be">'+ev.text.replace(/</g,'&lt;')+'</span>':'';
    row.innerHTML = lbl + body;
    log.appendChild(row); log.scrollTop = log.scrollHeight;
  }
  function connect(){
    var proto = location.protocol==='https:'?'wss':'ws';
    var ws = new WebSocket(proto+'://'+location.host+'/duet/events');
    ws.onopen=function(){dot.style.color='#54E0C7'};
    ws.onclose=function(){dot.style.color='#FF6A5A';setTimeout(connect,1500)};
    ws.onmessage=function(e){try{add(JSON.parse(e.data))}catch(_){}}
  }
  connect();
})();
"""


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=PORT)
