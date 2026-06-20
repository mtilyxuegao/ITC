"""用户语音 ASR:把上行音频(float32 16kHz PCM)转写成文字,喂给大模型。
用 OpenAI 转写接口(复用 OPENAI_TOKEN),不自建 Whisper。
"""
from __future__ import annotations

import array
import io
import logging
import sys
import wave

import aiohttp

logger = logging.getLogger("tt.asr")


def pcm_f32_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """float32 little-endian PCM → 16-bit WAV 字节(纯标准库,不依赖 numpy)。"""
    n = (len(pcm_bytes) // 4) * 4
    f = array.array("f")
    f.frombytes(pcm_bytes[:n])
    if sys.byteorder == "big":
        f.byteswap()  # 输入按 little-endian 解释
    h = array.array("h", (max(-32768, min(32767, int(x * 32767.0))) for x in f))
    if sys.byteorder == "big":
        h.byteswap()  # WAV 需 little-endian
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(h.tobytes())
    return buf.getvalue()


async def transcribe(session: aiohttp.ClientSession, api_key: str, pcm_bytes: bytes,
                     base_url: str = "https://api.openai.com/v1",
                     model: str = "gpt-4o-mini-transcribe",
                     language: str = "zh", sample_rate: int = 16000) -> str:
    """转写一段音频,返回文字。失败返回 ""。"""
    if len(pcm_bytes) < sample_rate * 4 // 2:  # < 0.5s,太短不转
        return ""
    wav = pcm_f32_to_wav(pcm_bytes, sample_rate)
    form = aiohttp.FormData()
    form.add_field("model", model)
    form.add_field("language", language)
    form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}  # local 无需鉴权
    url = base_url.rstrip("/") + "/audio/transcriptions"
    try:
        async with session.post(url, data=form, headers=headers) as resp:
            if resp.status >= 400:
                logger.warning("asr error %s: %s", resp.status, (await resp.text())[:200])
                return ""
            data = await resp.json()
        return (data.get("text") or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("asr failed: %s", e)
        return ""
