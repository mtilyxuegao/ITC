"""The control-token event log — the demo hero and the test oracle.

Every meaningful thing the Conductor does emits an Event. The UI renders these as
a live, color-coded timeline (`^ · [CUT] · [THINK#2] · [WAIT] · inject→speak`);
the tests assert on the sequence and epochs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional


class Kind:
    """Control-token / event kinds. String constants keep the UI JSON trivial."""
    ONSET = "^"                 # user voice onset while model speaks (logged only)
    CUT = "[CUT]"               # true barge-in: model audio actually stops
    THINK = "[THINK]"           # dispatch async work to the thinking layer
    WAIT = "[WAIT]"             # suspend/reset in-flight thinking on new info
    PENDNS = "[PENDNS]"         # mutual silence window
    DISPATCH = "dispatch"       # thinking task spawned (carries the request)
    MILESTONE = "inject→speak"  # a thinking-layer partial, injected & spoken
    RESULT = "RESULT"           # final answer injected & spoken
    DROPPED = "dropped"         # a stale-epoch event killed by the epoch guard
    SPEAK = "speak"             # model spoke (generic, e.g. immediate ack)
    USER = "user"               # a user turn entered the dialog


@dataclass
class Event:
    kind: str
    epoch: int
    text: str = ""
    ts: float = field(default_factory=time.monotonic)
    meta: dict = field(default_factory=dict)

    def label(self) -> str:
        """Human/UI label, e.g. '[THINK#2]'."""
        if self.kind in (Kind.THINK, Kind.WAIT, Kind.DISPATCH, Kind.RESULT, Kind.MILESTONE):
            return f"{self.kind}#{self.epoch}"
        return self.kind


class EventLog:
    """Append-only log with optional live subscribers (for the UI WebSocket feed)."""

    def __init__(self) -> None:
        self._events: List[Event] = []
        self._t0: Optional[float] = None
        self._subscribers: List[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._subscribers.append(fn)

    def emit(self, kind: str, epoch: int, text: str = "", **meta) -> Event:
        ev = Event(kind=kind, epoch=epoch, text=text, meta=meta)
        if self._t0 is None:
            self._t0 = ev.ts
        self._events.append(ev)
        for fn in self._subscribers:
            try:
                fn(ev)
            except Exception:
                # a flaky UI subscriber must never break the control loop
                pass
        return ev

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def kinds(self) -> List[str]:
        return [e.kind for e in self._events]

    def of_kind(self, kind: str) -> List[Event]:
        return [e for e in self._events if e.kind == kind]

    def to_json(self) -> str:
        """Serialize for the UI timeline (ms offsets from first event)."""
        t0 = self._t0 or 0.0
        rows = []
        for e in self._events:
            row = asdict(e)
            row["label"] = e.label()
            row["ms"] = round((e.ts - t0) * 1000, 1)
            rows.append(row)
        return json.dumps(rows, ensure_ascii=False)
