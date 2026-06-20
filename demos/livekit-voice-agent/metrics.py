"""Lightweight observability (THINKER_TALKER.md §8 item 7).

Monotonic counters + think-latency timing, logged via the stdlib logger.
One Metrics instance per session, owned by the agent. Trivially swappable for
Prometheus later.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Metrics:
    turns: int = 0
    escalations: int = 0
    interrupts: int = 0
    invalidated_results: int = 0
    thinker_done: int = 0
    thinker_timeouts: int = 0
    _think_latencies: list[float] = field(default_factory=list)

    def on_turn(self) -> None:
        self.turns += 1

    def on_escalate(self) -> None:
        self.escalations += 1

    def on_interrupt(self) -> None:
        self.interrupts += 1

    def on_invalidated(self) -> None:
        self.invalidated_results += 1

    def on_thinker_done(self) -> None:
        self.thinker_done += 1

    def on_thinker_timeout(self) -> None:
        self.thinker_timeouts += 1

    @contextmanager
    def time_think(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._think_latencies.append(time.perf_counter() - start)

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.turns if self.turns else 0.0

    @property
    def avg_think_latency(self) -> float:
        return sum(self._think_latencies) / len(self._think_latencies) if self._think_latencies else 0.0

    def snapshot(self) -> dict:
        return {
            "turns": self.turns,
            "escalations": self.escalations,
            "escalation_rate": round(self.escalation_rate, 3),
            "interrupts": self.interrupts,
            "invalidated_results": self.invalidated_results,
            "thinker_done": self.thinker_done,
            "thinker_timeouts": self.thinker_timeouts,
            "avg_think_latency_s": round(self.avg_think_latency, 2),
        }

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info("session metrics: %s", self.snapshot())
