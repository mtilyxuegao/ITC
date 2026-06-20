"""Piper TTS for the write-back seam.

The duplex model only ingests AUDIO. To make it voice the backend's result, we TTS
the result to 16kHz mono float32 PCM and inject it as input.append audio — the model
"hears" it and relays it to the user in its own voice.
"""
from __future__ import annotations

import os

import numpy as np

_VOICE_PATH = os.environ.get("PIPER_VOICE", os.path.expanduser("~/en_US-lessac-low.onnx"))
_voice = None


def _get_voice():
    global _voice
    if _voice is None:
        from piper import PiperVoice
        _voice = PiperVoice.load(_VOICE_PATH)
    return _voice


def tts_pcm_f32(text: str) -> bytes:
    """Synthesize text -> 16kHz mono float32 PCM bytes (matches MiniCPM's audio input)."""
    text = (text or "").strip()
    if not text:
        return b""
    parts = [chunk.audio_float_array for chunk in _get_voice().synthesize(text)]
    if not parts:
        return b""
    return np.concatenate(parts).astype(np.float32).tobytes()
