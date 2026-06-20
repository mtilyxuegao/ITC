"""GPU ASR service — faster-whisper large-v3 on CUDA.

The reliable READ seam: the proxy POSTs the user's (listen-phase, echo-free) audio
here and gets back an accurate transcript of the ACTUAL question — instead of trying
to reverse-engineer it from the frozen model's unreliable text.

  POST /asr   {"audio_b64": <base64 of 16kHz mono float32 PCM>, "language": null}
              -> {"text": "..."}
  GET  /health -> {"status": "ready"}
"""
from __future__ import annotations

import base64
import os

import numpy as np
from aiohttp import web

MODEL_NAME = os.environ.get("ASR_MODEL", "large-v3")
PORT = int(os.environ.get("ASR_PORT", "8020"))
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE") or None   # e.g. "en" to lock English Call
RMS_GATE = float(os.environ.get("ASR_RMS_GATE", "0.006"))
_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
    return _model


async def asr(request: web.Request) -> web.Response:
    body = await request.json()
    pcm = base64.b64decode(body.get("audio_b64", ""))
    if not pcm:
        return web.json_response({"text": ""})
    audio = np.frombuffer(pcm, dtype=np.float32)
    if audio.size < int(16000 * 0.3):
        return web.json_response({"text": ""})
    # energy gate: whisper hallucinates words on silence/near-silence -> skip those
    if float(np.sqrt(np.mean(np.square(audio)))) < RMS_GATE:
        return web.json_response({"text": ""})
    segments, _ = get_model().transcribe(
        audio, beam_size=1,   # greedy: ~2-3x faster on large-v3 for short voice turns, negligible WER
        language=body.get("language") or ASR_LANGUAGE,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        no_speech_threshold=0.6, condition_on_previous_text=False, temperature=0.0)
    text = " ".join(s.text for s in segments).strip()
    return web.json_response({"text": text})


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ready"})


def main():
    get_model()  # warm-load before serving so the first request is fast
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/asr", asr)
    app.router.add_get("/health", health)
    web.run_app(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
