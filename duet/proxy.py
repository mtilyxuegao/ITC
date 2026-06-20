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
from .direct_search import DirectSearchClient
from .events import Event, EventLog, Kind
from .hermes_thinking import HermesThinkingClient  # kept for fallback / reference
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
        self._cue_epoch = 0            # epoch whose progress-cue already fired (once per research)
        self._result_epoch = 0         # latest epoch a RESULT was seen for (suppress a late cue)
        self.session = None            # shared aiohttp session (set at startup; pooled keep-alive)
        self.recent: list[str] = []    # recent STT transcripts (context for the router)
        self.log = EventLog()
        self.log.subscribe(self._on_event)
        # FAST research path: ddgs (the ~1-2s bottleneck) + ONE LLM summarize (~120ms), skipping
        # hermes's decide-hop + agent-framework overhead (the router already produced the query).
        # ~3-6s -> ~1.5-2.2s. Swap back to HermesThinkingClient(THINK_URL, model=MODEL,
        # enabled_toolsets=TOOLSETS, max_iterations=3) if multi-step agentic research is needed.
        self.conductor = Conductor(
            thinking=DirectSearchClient(THINK_URL, model=MODEL),
            speaker=RecordingSpeaker(), log=self.log, scene="duplex")

    def _on_event(self, ev: Event) -> None:
        data = _event_json(ev)
        for q in list(self.subscribers):
            try:
                q.put_nowait(data)
            except Exception:
                pass
        if ev.kind == Kind.RESULT:
            self._result_epoch = max(self._result_epoch, ev.epoch)   # block any late cue for this epoch
            if ev.text:                              # write the answer back into the model's voice
                asyncio.create_task(self._voice_back(ev.text))
        elif ev.kind == Kind.MILESTONE:
            # FIRST milestone of a fresh, still-current research epoch -> voice a fixed progress cue
            # so the decoupling is AUDIBLE and the research dead-air is filled. Guards: once per epoch
            # (_cue_epoch), only the current epoch (not a stale/[WAIT]'d one), still THINKING, and no
            # RESULT already seen for it (_result_epoch).
            cur = self.conductor.epoch.current
            if (ev.epoch == cur and ev.epoch > self._cue_epoch and ev.epoch > self._result_epoch
                    and self.conductor.phase == Phase.THINKING):
                self._cue_epoch = ev.epoch
                asyncio.create_task(self._voice_cue(ev.epoch))

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
                await asyncio.sleep(8)   # research completes in <7s; never strand the model muted
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

    # Fixed canned cue — NEVER model output, so it can't garble through the decoder/TTS.
    CUE_TEXT = "Let me look that up."
    # Time the cue needs to be spoken before we re-mute; re-muting earlier feeds <|turn_eos|>
    # and clips the cue. ~5 words of TTS speech finishes well under this.
    CUE_SPEAK_SECS = float(os.environ.get("CUE_SPEAK_SECS", "2.5"))

    async def _voice_cue(self, epoch: int) -> None:
        """Voice a FIXED progress cue at the start of a research epoch via the SAME direct
        control.force_speak path as _voice_back. The model is muted (force_listen) during
        research, so clear the gate FIRST or the next browser input.append re-mutes mid-cue
        and clips it; then RE-MUTE after the cue is spoken so anti-抢答 is restored. The re-mute
        is guarded (epoch+phase+_result_epoch+same-connection) so a RESULT or newer dispatch
        during the cue is never clobbered — no double-talk, no stranded mute."""
        up = self.active_up
        if up is None or getattr(up, "closed", True):
            return
        if (self.conductor.phase != Phase.THINKING or epoch != self.conductor.epoch.current
                or epoch <= self._result_epoch):
            return
        self._set_listen_gate(False)                 # release the model BEFORE forcing the cue
        gate = self.active_gate                       # snapshot the connection we cleared
        try:
            await up.send_str(json.dumps(
                {"type": "control.force_speak", "payload": {"text": self.CUE_TEXT}},
                ensure_ascii=False))
            print(f"[duet] cue force_speak (epoch {epoch}) -> {self.CUE_TEXT!r}", flush=True)
        except Exception as e:
            print(f"[duet] cue force_speak failed: {e}", flush=True)
            return
        await asyncio.sleep(self.CUE_SPEAK_SECS)
        if (self.active_gate is gate and gate is not None
                and self.conductor.phase == Phase.THINKING
                and epoch == self.conductor.epoch.current
                and epoch > self._result_epoch):
            gate["force_listen"] = True
            print(f"[duet] cue done -> re-muted (epoch {epoch})", flush=True)

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
        # hermes already emits ONE short spoken sentence with worded numbers (see its prompt),
        # so just strip markdown/trim locally — NO extra gemma spokenify hop on the critical path.
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
    async def _route(self, transcript: str, context: list, frame_b64: str | None = None):
        """Context-aware query builder: resolve the LATEST turn into ONE standalone web query,
        or NONE for chit-chat. If a camera frame_b64 (JPEG, no data: prefix) is given, attach it
        to the gemma (vision-enabled, AUX_MODEL) call so deictic turns ('what is THIS', 'how much
        does this cost') are grounded in what the user is pointing at. Async path only."""
        ctx = " | ".join(context[-5:]) if context else "(none)"
        user_text = f"Recent turns: {ctx}\nLatest: {transcript}"
        if frame_b64:
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + frame_b64}}]
        else:
            user_content = user_text
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
                        "stock price'), keep them BOTH in one query — do NOT drop it; (4) if an "
                        "IMAGE is attached and the latest uses a pointing word ('this', 'that', "
                        "'here', 'it') with no clear referent, IDENTIFY the main object in the "
                        "image and put its concrete name in the query (e.g. with a power bank in "
                        "view, 'how much does this cost' -> 'Anker power bank price'). Output the "
                        "query for ANY question or information request, however casual. Output "
                        "exactly NONE ONLY when the latest is purely a greeting, thanks, "
                        "acknowledgement, or filler with literally no question in it."},
                       {"role": "user", "content": user_content}]}
        try:
            async with self.session.post(AUX_URL + "/v1/chat/completions", json=payload,
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
            async with self.session.post(ASR_URL + "/asr",
                                          json={"audio_b64": base64.b64encode(pcm).decode()},
                                          timeout=aiohttp.ClientTimeout(total=30)) as r:
                d = await r.json()
            return (d.get("text") or "").strip()
        except Exception as e:
            print(f"[duet] ASR error: {e}", flush=True)
            return ""

    async def on_user_audio(self, pcm: bytes, frame_b64: str | None = None) -> None:
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
        query = await self._route(transcript, self.recent, frame_b64)   # vision-grounded if a frame
        _ms_route = (loop.time() - _t1) * 1000
        if not query:
            self._set_listen_gate(False)          # chit-chat -> release the optimistic mute
            print(f"[duet] route: drop (route {_ms_route:.0f}ms) -> {transcript[:50]!r}", flush=True)
            return
        self._t_dispatch = loop.time()            # for end-to-end answer-latency timing in _voice_back
        print(f"[duet] DISPATCH (route {_ms_route:.0f}ms) -> {query!r}", flush=True)
        self._set_listen_gate(True)               # mute the model during research (anti-抢答)
        tok = self._gate_token                    # to release this mute the instant research ENDS
        if self.conductor.phase == Phase.THINKING:
            await self.conductor.on_interrupt()   # new turn supersedes -> [WAIT]
        await self.conductor.on_user_utterance(query)
        think = self.conductor._think_task
        if think is not None:
            async def _release_when_done(task=think, tk=tok):
                try:
                    await task
                except Exception:
                    pass
                # research ended — success voices via _voice_back (which bumps the token, so this
                # is then a no-op); this covers the fail/empty/cancel path so the model un-mutes
                # immediately instead of waiting out the 8s failsafe. Token guard = don't clobber
                # a newer dispatch's mute.
                if self._gate_token == tk and self.active_gate is not None and self.active_gate.get("force_listen"):
                    self.active_gate["force_listen"] = False
                    print("[duet] gate released (research ended)", flush=True)
            asyncio.create_task(_release_when_done())


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

    loop = asyncio.get_event_loop()
    user_pcm = bytearray()       # USER mic audio this turn (16kHz mono f32), echo-free
    state = {"speaking": False, "force_listen": False, "last_audio_t": 0.0, "last_frame": None}
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
                inp = msg.get("input") or {}
                a = inp.get("audio")
                if a:
                    try:
                        user_pcm.extend(base64.b64decode(a))
                        state["last_audio_t"] = loop.time()   # for silence-based turn detection
                    except Exception:
                        pass
                vf = inp.get("video_frames")              # MiniCPM omni sends camera frames here;
                if vf:
                    state["last_frame"] = vf[-1]          # keep the latest for the vision router
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
            # Self-recovery: if the model is just listening and we are NOT mid-research, the gate
            # has no reason to stay set — release it so a stuck force_listen can never strand the
            # model muted ("can't continue" after an interrupt). During research the conductor is
            # THINKING, so the anti-抢答 mute is preserved.
            if state["force_listen"] and hub.conductor.phase != Phase.THINKING:
                state["force_listen"] = False
                print("[duet] gate released on listen (not researching)", flush=True)

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

            async def silence_dispatcher():
                # Turn-end = the USER paused (silence), NOT the model starting to speak — so a
                # muted/silent model can never stall turn detection ("can't continue after interrupt").
                # On a >0.6s pause after >=0.3s of speech, grab the turn, arm the anti-抢答 mute, dispatch.
                while True:
                    await asyncio.sleep(0.15)
                    if state["speaking"]:
                        continue
                    if len(user_pcm) > int(16000 * 4 * 0.3) and (loop.time() - state["last_audio_t"]) > 0.6:
                        pcm = bytes(user_pcm); user_pcm.clear()
                        frame = state["last_frame"]              # latest camera frame this turn (or None)
                        async def _arm(st=state):                # anti-抢答 mute after a short ack grace
                            await asyncio.sleep(GATE_DELAY)
                            st["force_listen"] = True
                        asyncio.create_task(_arm())
                        asyncio.create_task(hub.on_user_audio(pcm, frame))

            sd = asyncio.create_task(silence_dispatcher())
            try:
                await asyncio.gather(c2u(), u2c())
            finally:
                sd.cancel()
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
        app["hub"].session = app["session"]   # pooled keep-alive for ASR/router calls

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
