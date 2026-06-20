"""In-process SessionState + epoch fencing (THINKER_TALKER.md §2.1, §8 item 1).

This is the "Blackboard" the Talker and Thinker bridge through. It is pure data
plus a few helpers — no LiveKit or HTTP imports — so it unit-tests trivially.

Concurrency assumption: everything runs on ONE asyncio event loop (single box),
so no locks are needed. The Thinker streams in a task on the SAME loop and only
*reads* `epoch` via `is_current()`. When this moves cross-box (Redis), `epoch`
becomes an optimistic-concurrency / fencing token and writes need a CAS.
"""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field


class TalkerState(str, enum.Enum):
    LISTENING = "LISTENING"     # idle / perceiving
    SPEAKING = "SPEAKING"       # emitting TTS (still interruptible)
    THINKING = "THINKING"       # escalated, awaiting the Thinker (filler plays)
    INTERRUPTED = "INTERRUPTED"  # barge-in detected, snapping back to LISTENING


class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    CANCELLED = "CANCELLED"


@dataclass
class Turn:
    role: str          # "user" | "assistant"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class ThinkerTask:
    task_id: str
    epoch: int          # ⭐ epoch snapshot at submit time, checked on write-back
    query: str
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None


@dataclass
class SessionState:
    session_id: str
    epoch: int = 0                                  # ⭐ monotonic fence; +1 on barge-in / re-escalation
    talker_state: TalkerState = TalkerState.LISTENING
    user_speaking: bool = False
    transcript: list[Turn] = field(default_factory=list)
    context_summary: str = ""
    thinker_task: ThinkerTask | None = None

    # --- epoch (the master switch for all cancellation) ---
    def bump_epoch(self) -> int:
        """Advance the fence. Every old-epoch result/task is now invalid."""
        self.epoch += 1
        return self.epoch

    def is_current(self, epoch: int) -> bool:
        """THE gate predicate, reused by the Thinker (cooperative abort) and the
        write-back gate (invalidate-at-sink)."""
        return epoch == self.epoch

    # --- single-active-task invariant (§3.5: at most one thinker_task) ---
    def start_task(self, query: str, epoch_snapshot: int) -> ThinkerTask:
        """Caller MUST have already cancelled+cleared any prior task."""
        task = ThinkerTask(task_id=_new_task_id(), epoch=epoch_snapshot, query=query)
        self.thinker_task = task
        return task

    def clear_task(self) -> None:
        self.thinker_task = None

    @property
    def has_active_task(self) -> bool:
        return self.thinker_task is not None and self.thinker_task.status in (
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
        )

    # --- transcript / query building ---
    def append_turn(self, role: str, text: str) -> None:
        if text:
            self.transcript.append(Turn(role=role, text=text))

    def build_query(self, max_turns: int = 8) -> str:
        """Compress recent transcript + summary into the Thinker prompt (§2.3:
        pass only compressed context, never raw audio/video or the whole KV)."""
        lines: list[str] = []
        if self.context_summary:
            lines.append(f"Conversation summary: {self.context_summary}")
        recent = self.transcript[-max_turns:]
        for turn in recent:
            speaker = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{speaker}: {turn.text}")
        return "\n".join(lines)


def _new_task_id() -> str:
    return f"think_{uuid.uuid4().hex[:12]}"
