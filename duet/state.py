"""Conductor state: the phase enum, the task state, and the epoch guard.

The EPOCH GUARD is the correctness backbone of [WAIT]. Every dispatch belongs to
an epoch; a user interrupt bumps the epoch. Anything (a streamed milestone, a late
result) tagged with a stale epoch is dropped. This makes interruption correct even
if vLLM keeps generating for a moment after we close the stream — we never trust
"GPU freed" as the correctness mechanism; that's only an efficiency bonus.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class Phase(enum.Enum):
    LISTENING = "listening"   # model live; conductor STT + VAD running
    THINKING = "thinking"     # async work dispatched; model stays live


@dataclass
class TaskState:
    """What the thinking layer needs to know. Passed on every [THINK]."""
    scene: str = "general"
    constraints: Dict[str, str] = field(default_factory=dict)  # e.g. origin/dest/date
    dialog: List[str] = field(default_factory=list)            # recent user turns
    notes: List[str] = field(default_factory=list)

    def apply_delta(self, delta: Dict[str, str]) -> None:
        for k, v in delta.items():
            if v is not None:
                self.constraints[k] = v

    def add_user_turn(self, text: str) -> None:
        self.dialog.append(text)

    def summary(self) -> str:
        c = self.constraints
        parts = [f"{k}={v}" for k, v in c.items()]
        return f"scene={self.scene} " + " ".join(parts)

    def snapshot(self) -> dict:
        return {
            "scene": self.scene,
            "constraints": dict(self.constraints),
            "dialog": list(self.dialog),
        }


class EpochManager:
    """Monotonic epoch counter. bump() on every [WAIT]/interrupt."""

    def __init__(self, start: int = 1) -> None:
        self._epoch = start

    @property
    def current(self) -> int:
        return self._epoch

    def bump(self) -> int:
        self._epoch += 1
        return self._epoch

    def is_current(self, epoch: int) -> bool:
        return epoch == self._epoch
