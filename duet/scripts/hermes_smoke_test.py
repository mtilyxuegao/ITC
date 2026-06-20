#!/usr/bin/env python3
"""Validate HermesThinkingClient end-to-end against a LIVE vLLM thinking-layer.

Proves the thinking-layer half of DUET-Lite works through the hermes harness:
  T1  run a task -> hermes drives the agent loop on our vLLM -> milestones + RESULT
  T2  cancel mid-run -> agent.interrupt() fires ([WAIT] path) -> aborts recorded

  PYTHONPATH=/home/justin/ITC:/home/justin/hermes-agent \
  /home/justin/hermes-env/bin/python -m duet.scripts.hermes_smoke_test \
      --base-url http://liquid-gpu-001:8001 --model qwen
"""
from __future__ import annotations

import argparse
import asyncio

from duet.hermes_thinking import HermesThinkingClient
from duet.state import TaskState
from duet.thinking import RESULT


async def main(args) -> int:
    toolsets = [t for t in args.toolsets.split(",") if t]
    client = HermesThinkingClient(args.base_url, model=args.model, enabled_toolsets=toolsets)

    print(f"=== T1: hermes run @ {args.base_url} model={args.model} toolsets={toolsets} ===")
    emits = []

    async def emit(kind, epoch, text):
        emits.append((kind, text))
        print(f"  [{kind}] {text[:110]}")

    await client.run(1, TaskState(constraints={"origin": "PVG", "date": "Fri"}), emit, lambda: True)
    ok1 = any(k == RESULT for k, _ in emits)
    print(f"[T1] {'PASS' if ok1 else 'FAIL'} -- hermes produced a RESULT via vLLM")

    print("\n=== T2: cancel mid-run -> interrupt() ([WAIT] path) ===")

    async def emit2(kind, epoch, text):
        pass

    t = asyncio.create_task(client.run(2, TaskState(constraints={"origin": "PEK"}), emit2, lambda: True))
    await asyncio.sleep(1.5)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    ok2 = client.aborts >= 1
    print(f"[T2] aborts={client.aborts} {'PASS' if ok2 else 'FAIL'} -- interrupt fired on cancel")

    print("-" * 56)
    print("RESULT:", "ALL PASS" if (ok1 and ok2) else "SOME FAILED")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="qwen")
    p.add_argument("--toolsets", default="", help="comma-separated hermes toolsets, e.g. 'web'")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args)))
