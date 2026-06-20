# DUET-Lite — overnight optimization pass (for morning review)

Reverted the research model back to **Qwen3.5-35B-A3B** as requested, then applied the
**safe, self-testable** subset of the optimization roadmap and deferred the rest (anything
that changes the *sound* of the output, or carries demo-break risk, needs your ears).

## Current live state (self-tested, no voice)
- research = **Qwen3.5-35B-A3B** (`qwen`, liquid-gpu-026:8001) — tool-calling ✓, ~163+ tok/s
- aux = **gemma-4-12B-it** (`gemma`, liquid-gpu-026:8002) — router ✓ (compound query kept, chit-chat dropped)
- interaction = **MiniCPM-o duplex + proxy + ngrok** (liquid-gpu-024) — `/omni` 200, force_speak smoke ✓
- ASR = whisper-large-v3 (liquid-gpu-037)
- public URL (unchanged): https://duet-omni.openclaw-claw-train.ngrok.app/omni
- **hard-refresh, use ONE tab.**

## APPLIED + verified (committed)
1. **Content-based 抢答 gate** (proxy.py tap_gateway) — the single highest-leverage fix. The
   model's `kind=text` delta carries the spoken words *before* that chunk's audio (verified), so
   the instant a fact token (digit / `$ % °`) appears past the ack, we mute. Cuts a hallucinated
   number/price *before it is heard*. **Additive** — the GATE_DELAY=1.0s timer remains the
   fallback, so this can only mute *earlier*, never worse. (Couldn't voice-test; mechanism verified.)
2. **Per-stage timing logs** (proxy.py) — `STT (asr Nms)`, `DISPATCH (route Nms)`,
   `force_speak (research+spokenify Nms)` in the proxy log, so we stop tuning blind. Watch
   `/home/justin/duet-proxy.<job>.log`.
3. **ASR beam_size 5→1** (asr_server.py) — greedy decode, ~2-3x faster on short turns, kept the
   VAD/RMS/no_speech hallucination guards.
4. **Zombie-worker watchdog** (`duet/deploy/watchdog.sh`, running pid in
   `/home/justin/duet-watchdog.log`) — auto-recovers the stuck duplex worker you kept hitting.
   CONSERVATIVE trigger so it never kills a live session: fires only when `busy>=1 & idle==0 &
   queue>=1 (someone blocked) & held >150s`, sustained 3 polls, 7-min cooldown. Dormant when
   nobody is connecting. Stop it: `kill $(pgrep -f duet-watchdog)`.

## DEFERRED — recommended, but needs your voice check / carries risk (NOT applied)
Ranked; each is a clear win, I just won't ship it un-heard while you sleep:
- **Kill the 2nd (critical-path) spokenify** (~-300-600ms, the most-felt latency): drop the
  gemma `_spokenify` call inside `_voice_back`, use local `_condense`, and move "spell numbers as
  words" into the hermes answer prompt. Deferred because it changes the spoken wording — verify by ear.
- **torch.compile ON + pre-warm** (~15-30% off MiniCPM TTFT): set `COMPILE_FLAG=--compile` in
  serve_interaction.sbatch + fire one dummy utterance at startup. Deferred: lazy first-forward
  JIT can hang turn-1 (we hit this before) — needs a verified pre-warm, best watched live.
- **Voice the first thinking milestone** as a canned "searching now" cue — makes the decoupling
  AUDIBLE (currently panel-only) and fills the dead-air gap. Deferred: timing race with the
  answer force_speak could double-talk; wants a voice run to tune.
- **Real voice barge-in → [CUT]/[WAIT] re-dispatch** ("interruptible both ways", the canonical
  DuplexOmni beat) — wire an energy-gated onset during `state["speaking"]` to the already-built
  `conductor.on_interrupt()`.
- **Camera frames → backend** (the omni money-shot; frames already arrive at the proxy; gemma-12B
  vision is enabled) and **fast-gemma + deep-hermes concurrent agents** (the "multiple background
  agents" goal). Both medium effort, voice/vision demo features for you to drive.
- pooled aiohttp session (~-50-250ms jitter), 2nd duplex worker (kill the single-worker SPOF).

## SKIP (not worth the 24h risk) — see the full roadmap
- jisen/thinker-talker continuous-loop port (large, model-initiated CUT on a frozen model is the
  exact 抢答 risk we tamed), token-streamed force_speak (needs new chunked injection), gemma
  audio-in to drop whisper (lowest demo upside, loses whisper's guards).

Full 5-dimension analysis + roadmap JSON: the workflow output in this session.
