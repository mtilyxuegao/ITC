"""Drive the Thinker (search + reasoning + dashboard events) WITHOUT the voice loop.

Lets you verify features 1 & 2 end-to-end with no OpenAI/LiveKit keys:
    python dashboard.py &                      # open http://localhost:8800
    DASHBOARD_URL=http://localhost:8800 python sim_thinker.py "what's the latest on <X>?"

It builds a fake escalated turn and runs ThinkerClient.run, so the dashboard shows
🧠 engaged → 🔍 searching → 📄 found → ✅ answer, and the conclusion prints here.
"""
from __future__ import annotations

import asyncio
import sys

from config import load_config
from session_state import SessionState
from writeback import ResultEventChannel
from metrics import Metrics
from thinker_client import ThinkerClient
import events

QUESTION = " ".join(sys.argv[1:]) or "What are the latest large language models released in 2026?"


async def main():
    cfg = load_config()
    state = SessionState(session_id="sim")
    state.append_turn("user", QUESTION)
    task = state.start_task(state.build_query(), state.epoch)

    # mirror the agent's escalation event so the dashboard shows the engage banner
    events.emit(cfg, "escalated", session_id=state.session_id, epoch=task.epoch,
                task_id=task.task_id, objective=QUESTION, reason="sim")

    channel = ResultEventChannel()
    metrics = Metrics()
    tc = ThinkerClient(cfg)
    run_task = asyncio.create_task(tc.run(task, state, channel, metrics, QUESTION))

    answer: list[str] = []
    async def consume():
        async for ev in channel.subscribe():
            if ev.is_fallback:
                print(">>> FALLBACK (timeout/error)")
                return
            if ev.sentence:
                answer.append(ev.sentence)
            if ev.is_final:
                return

    try:
        await asyncio.wait_for(consume(), timeout=cfg.timeout_s + 15)
    except asyncio.TimeoutError:
        print(">>> consume timed out")
    await run_task
    await asyncio.sleep(1.0)  # let fire-and-forget dashboard emits flush before exit

    print("\n========== QUESTION ==========")
    print(QUESTION)
    print("========== SPOKEN ANSWER ==========")
    print(" ".join(answer).strip() or "(empty)")
    print("===================================")


if __name__ == "__main__":
    asyncio.run(main())
