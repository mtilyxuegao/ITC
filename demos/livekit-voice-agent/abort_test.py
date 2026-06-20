"""Live verification of the Thinker endpoint + the barge-in abort mechanism.

Run on the server against a live vLLM Thinker:
    ~/thinker-venv/bin/python abort_test.py

Checks:
  1. /v1/models lists the served model.
  2. A reasoning completion returns, and the chain-of-thought is split into
     `reasoning_content` so `content` is ONLY the spoken conclusion (requires
     vLLM launched with --reasoning-parser deepseek_r1).
  3. THE CRITICAL TEST: start a long streaming completion, confirm the server
     reports 1 running request, cancel the asyncio task that owns the stream, and
     confirm the server's running-request count drops to 0 — proving cancel-at-source
     actually frees the GPU, exactly as thinker_client.ThinkerClient.abort relies on.
"""
from __future__ import annotations

import asyncio
import os
import re
import time

import httpx
from openai import AsyncOpenAI

BASE_URL = os.getenv("THINKER_BASE_URL", "http://localhost:8000/v1")
METRICS_URL = BASE_URL.rstrip("/").removesuffix("/v1") + "/metrics"
MODEL = os.getenv("THINKER_MODEL", "itc-thinker")

# Disable HTTP keep-alive so that closing a streaming response actually CLOSES the
# TCP socket (instead of returning it to the pool). That socket close is what makes
# vLLM see http.disconnect and abort the in-flight request (cancel-at-source).
_http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0))
client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY", http_client=_http_client)


async def running_requests() -> float:
    """Read vllm:num_requests_running from the Prometheus /metrics endpoint."""
    async with httpx.AsyncClient(timeout=5.0) as hc:
        r = await hc.get(METRICS_URL)
    for line in r.text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            m = re.search(r"\s([\d.]+)\s*$", line)
            if m:
                return float(m.group(1))
    return -1.0


async def smoke_models():
    models = await client.models.list()
    ids = [m.id for m in models.data]
    print(f"[1] MODELS: {ids}")
    assert MODEL in ids, f"{MODEL} not served!"


async def smoke_completion():
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 17% of 4230? Give just the number."}],
        max_tokens=2048,
        temperature=0.6,
    )
    dt = time.perf_counter() - t0
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    reasoning = getattr(msg, "reasoning_content", None) or ""
    print(f"[2] completion in {dt:.1f}s")
    print(f"    reasoning_content: {len(reasoning)} chars (hidden from user)")
    print(f"    content (spoken):  {len(content)} chars -> {content[:160]!r}")
    clean = "<think>" not in content and len(content) < 400
    print(f"    content is a clean conclusion (no CoT leak): {clean}")


async def _long_stream(counter: dict):
    async with await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user",
                   "content": "Reason in extreme detail and count slowly from 1 to 400, "
                              "writing a full sentence about each number."}],
        stream=True,
        max_tokens=4000,
        temperature=0.6,
    ) as stream:
        async for chunk in stream:
            ch = chunk.choices[0] if chunk.choices else None
            if ch and (ch.delta.content or getattr(ch.delta, "reasoning_content", None)):
                counter["chunks"] += 1


async def abort_test():
    counter = {"chunks": 0}
    task = asyncio.create_task(_long_stream(counter))
    await asyncio.sleep(2.5)  # let it stream a while
    before = await running_requests()
    print(f"[3] streamed ~{counter['chunks']} chunks; server running_requests={before} (expect 1.0)")
    t_cancel = time.perf_counter()
    task.cancel()
    try:
        await task
        print("    !! task completed instead of cancelling")
    except asyncio.CancelledError:
        print(f"    client task cancelled cleanly in {(time.perf_counter()-t_cancel)*1000:.0f}ms")
    # Give the server a moment to observe the disconnect and abort the request.
    await asyncio.sleep(3.0)
    after = await running_requests()
    print(f"    server running_requests AFTER cancel={after} (expect 0.0 => GPU freed)")
    verdict = "PASS — server aborted at source" if after == 0.0 else "CHECK — request may still be running"
    print(f"    >>> ABORT VERDICT: {verdict}")


async def main():
    await smoke_models()
    await smoke_completion()
    await abort_test()
    print("\nABORT_TEST_DONE")


if __name__ == "__main__":
    asyncio.run(main())
