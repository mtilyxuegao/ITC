"""Write-back channel + epoch gate (THINKER_TALKER.md §4.3, §5, §8 item 4).

The Thinker streams ResultEvents into an in-process asyncio.Queue; the Talker's
result-pump consumes them and runs each through WriteBackGate, which is the
"invalidate-at-sink" gate (gate 2 of the doc's double protection). A result whose
epoch no longer matches the session epoch is discarded and NEVER spoken.

Single-box now (asyncio.Queue). Cross-box swap point: replace the queue with
Redis Streams / pub-sub, routing by session_id (documented, deferred).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator

from session_state import SessionState

logger = logging.getLogger("thinker-talker.writeback")


@dataclass
class ResultEvent:
    task_id: str
    epoch: int            # epoch snapshot taken when the task was submitted
    sentence: str         # one sentence of the Thinker conclusion (streaming, §5)
    is_final: bool        # last event for this task
    is_fallback: bool = False  # set on timeout/error -> speak the fallback line


class ResultEventChannel:
    """Thin async pub-sub over a single queue (one consumer: the result-pump)."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[ResultEvent] = asyncio.Queue()

    def publish(self, evt: ResultEvent) -> None:
        self._q.put_nowait(evt)

    async def subscribe(self) -> AsyncIterator[ResultEvent]:
        while True:
            yield await self._q.get()


class WriteBackGate:
    """The §4.1-⑤ predicate: only let through results for the CURRENT epoch."""

    def accept(self, evt: ResultEvent, state: SessionState) -> bool:
        if state.is_current(evt.epoch):
            return True
        logger.info(
            "discarding stale Thinker result task=%s epoch=%s (current=%s)",
            evt.task_id, evt.epoch, state.epoch,
        )
        return False
