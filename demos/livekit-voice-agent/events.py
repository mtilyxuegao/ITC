"""Fire-and-forget event emitter for the Thinker Activity dashboard.

The agent and the Thinker call emit(cfg, type, ...) to surface what the "second
brain" is doing. Events are POSTed to the dashboard (config.dashboard_url) without
ever blocking or failing the real-time loop — if the dashboard is down, emits are
silently dropped. This is purely for visualization; it carries no control logic.

Event types (payload keys in parens):
  escalated       (objective, reason)      -> 🧠 Thinker engaged
  thinking        ()                        -> reasoning started
  search_start    (query)                   -> 🔍 searching
  search_results  (query, results[])        -> 📄 found
  answer          (text)                     -> ✅ conclusion ready
  timeout         ()                         -> ⏱ gave up, fell back
  interrupted     ()                         -> ✋ barge-in, result discarded
  spoken          (text)                     -> 🔊 Talker said it
"""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

logger = logging.getLogger("thinker-talker.events")


def emit(cfg, etype: str, *, session_id: str = "", epoch: int = -1,
         task_id: str = "", **payload) -> None:
    """Non-blocking: schedule a POST and return immediately."""
    if not cfg or not getattr(cfg, "dashboard_url", ""):
        return
    event = {
        "type": etype,
        "ts": time.time(),
        "session_id": session_id,
        "epoch": epoch,
        "task_id": task_id,
        **payload,
    }
    try:
        asyncio.get_running_loop().create_task(_post(cfg.dashboard_url, event))
    except RuntimeError:
        # no running loop (e.g. unit test) — just skip
        pass
    logger.info("event %s %s", etype, {k: v for k, v in payload.items() if k != "results"})


async def _post(base_url: str, event: dict) -> None:
    try:
        async with httpx.AsyncClient(timeout=2.0) as hc:
            await hc.post(base_url.rstrip("/") + "/emit", json=event)
    except Exception:
        pass  # dashboard down / unreachable — visualization is best-effort
