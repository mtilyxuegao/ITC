"""STT for the READ seam — transcribe the USER's mic audio.

The fix for "the thinking layer got the model's deflection instead of my question":
we transcribe the user's own audio (which the proxy already relays as input.append
16 kHz mono float32 PCM) so the thinking layer dispatches on the USER's actual words.

faster-whisper "base" on CPU is ~real-time for short turns. Lazy-loaded so importing
this module is cheap.
"""
from __future__ import annotations

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe_pcm_f32(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe 16 kHz mono float32 PCM bytes. Returns '' for too-short/silent."""
    if not pcm_bytes:
        return ""
    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
    if audio.size < int(sample_rate * 0.4):   # < 0.4s -> skip
        return ""
    segments, _ = _get_model().transcribe(audio, vad_filter=True)
    return " ".join(s.text for s in segments).strip()
