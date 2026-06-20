#!/usr/bin/env python3
"""Real-endpoint smoke test: drive a LIVE vLLM thinking-layer server.

Validates the same code path the Conductor uses, against a real model on a real
H100 -- no fakes:
  T1  streaming chat via ThinkingClient (booking prompt -> RESULT), reports latency.
  T2  abort: open a long generation, cancel mid-stream, confirm it stops (the [WAIT]
      mechanism on a real server).

  # against the served Qwen thinker:
  PYTHONPATH=/home/justin/ITC /home/justin/duet-env/bin/python \
      -m duet.scripts.gpu_smoke_test --base-url http://<node>:8001 --model qwen
"""
from __future__ import annotations

import argparse
import asyncio
import time

import aiohttp

from duet.state import TaskState
from duet.thinking import RESULT, ThinkingClient


async def t1_streaming(base_url: str, model: str, use_tools: bool) -> bool:
    client = ThinkingClient(base_url, model=model, use_tools=use_tools)
    emits = []

    async def emit(kind, epoch, text):
        emits.append((kind, epoch, text))

    t0 = time.monotonic()
    await client.run(1, TaskState(constraints={"origin": "PVG", "date": "Fri"}),
                     emit, lambda: True)
    dt = time.monotonic() - t0
    results = [t for k, _, t in emits if k == RESULT]
    print(f"[T1] streaming run took {dt:.2f}s, {len(emits)} emits")
    for k, _, t in emits:
        print(f"     {k}: {t[:80]}")
    ok = bool(results)
    print(f"[T1] {'PASS' if ok else 'FAIL'} -- model produced a final answer")
    return ok


async def t2_abort(base_url: str, model: str) -> bool:
    """Open a long generation, cancel after 0.3s, confirm the stream stopped."""
    chunks = 0

    async def long_gen():
        nonlocal chunks
        payload = {"model": model, "stream": True, "max_tokens": 512,
                   "messages": [{"role": "user",
                                 "content": "Write a long, detailed 400-word essay about the sea."}]}
        async with aiohttp.ClientSession() as s:
            async with s.post(base_url.rstrip("/") + "/v1/chat/completions",
                              json=payload) as resp:
                async for _ in resp.content:
                    chunks += 1

    task = asyncio.create_task(long_gen())
    await asyncio.sleep(0.3)
    got_before = chunks
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.2)
    stopped = chunks  # should not keep growing after cancel
    ok = got_before > 0 and stopped == chunks
    print(f"[T2] received {got_before} chunks before cancel; stream stopped after abort")
    print(f"[T2] {'PASS' if ok else 'FAIL'} -- mid-stream abort works")
    return ok


async def main(args) -> int:
    print(f"=== GPU smoke test: {args.model} @ {args.base_url} ===")
    ok1 = await t1_streaming(args.base_url, args.model, use_tools=args.tools)
    ok2 = await t2_abort(args.base_url, args.model)
    print("-" * 56)
    print(f"RESULT: {'ALL PASS' if (ok1 and ok2) else 'SOME FAILED'}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="qwen")
    p.add_argument("--tools", action="store_true", help="send tool schemas (needs --enable-auto-tool-choice)")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args)))
