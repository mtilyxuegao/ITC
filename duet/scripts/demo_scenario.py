#!/usr/bin/env python3
"""Run the full decoupled scenario against the in-process fakes and print the
control-token timeline -- a text rendering of the demo's "hero" UI.

  python -m duet.scripts.demo_scenario

No GPU needed: a FakeVLLM stands in for the thinking layer and a RecordingSpeaker
stands in for the model's voice. Swap in real endpoints to go live (see README).
"""
from __future__ import annotations

import asyncio

from duet.conductor import Conductor
from duet.fakes import FakeVLLM
from duet.thinking import ThinkingClient
from duet.write_seam import RecordingSpeaker

_COLOR = {
    "[THINK]": "\033[36m", "[WAIT]": "\033[31m", "[CUT]": "\033[31m",
    "^": "\033[33m", "RESULT": "\033[32m", "inject→speak": "\033[32m",
    "user": "\033[33m", "dropped": "\033[90m",
}
_RESET = "\033[0m"


async def main() -> None:
    async with FakeVLLM(chunk_delay=0.05) as fv:
        speaker = RecordingSpeaker()
        c = Conductor(thinking=ThinkingClient(fv.base_url), speaker=speaker)

        def show(ev):
            color = _COLOR.get(ev.kind, "")
            print(f"  {ev.label():<14} epoch={ev.epoch}  {color}{ev.text}{_RESET}")

        c.log.subscribe(show)

        print("\n=== DUET-Lite scenario: book flight -> barge-in -> reset ===\n")
        print("USER: 订周五上海出发最便宜的航班")
        await c.on_user_utterance("订周五上海出发最便宜的航班")
        await asyncio.sleep(0.12)
        print("\nUSER (barge-in): 等下，改成北京出发\n")
        await c.on_user_utterance("等下，改成北京出发")
        await c.drain(timeout=5.0)

        print("\n--- what the user heard (in order) ---")
        for epoch, text in speaker.spoken:
            print(f"  [epoch {epoch}] {text}")
        print(f"\nfinal phase: {c.phase.value}, final epoch: {c.epoch.current}")


if __name__ == "__main__":
    asyncio.run(main())
