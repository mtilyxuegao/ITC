"""Intent classification over the Conductor's OWN streaming-STT transcript.

This is the READ seam: we never tap MiniCPM's internal text stream. We run our own
STT on a tee of the mic; this cheap, fast transcript is what fires [THINK]/[WAIT].

The classifier is deliberately heuristic (regex + keyword + a tiny city gazetteer).
For the hackathon that's enough and it's debuggable under pressure; swap in an
embedding model or a 1.7B classifier later behind the same `classify()` signature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from .state import TaskState

# knowledge intents that warrant dispatching the thinking layer
_KNOWLEDGE = [
    "订", "查", "找", "搜", "搜索", "航班", "机票", "天气", "预订", "帮我",
    "book", "find", "search", "flight", "weather", "look up", "lookup", "check",
]
# markers that a turn is correcting / interrupting with new info
_INTERRUPT = [
    "等下", "等等", "等一下", "其实", "算了", "不是", "改成", "改为", "改到", "换成",
    "wait", "actually", "instead", "no,", "no ", "hold on", "change", "scratch that",
]
# departure / destination markers
_DEPART = ["出发", "从", "起飞", "depart", "from", "leaving", "leave from"]
_DEST = ["到", "去", "飞往", "前往", "to ", "->"]

# tiny gazetteer: surface form -> (airport code, canonical name)
_CITIES = {
    "上海": ("PVG", "上海"), "浦东": ("PVG", "上海"), "虹桥": ("SHA", "上海"),
    "北京": ("PEK", "北京"), "首都": ("PEK", "北京"), "大兴": ("PKX", "北京"),
    "广州": ("CAN", "广州"), "深圳": ("SZX", "深圳"), "杭州": ("HGH", "杭州"),
    "成都": ("CTU", "成都"), "洛杉矶": ("LAX", "洛杉矶"), "纽约": ("JFK", "纽约"),
    "shanghai": ("PVG", "Shanghai"), "beijing": ("PEK", "Beijing"),
    "los angeles": ("LAX", "Los Angeles"), "new york": ("JFK", "New York"),
}
_WEEKDAYS = {
    "周一": "Mon", "周二": "Tue", "周三": "Wed", "周四": "Thu", "周五": "Fri",
    "周六": "Sat", "周日": "Sun", "今天": "today", "明天": "tomorrow",
    "monday": "Mon", "tuesday": "Tue", "wednesday": "Wed", "thursday": "Thu",
    "friday": "Fri", "saturday": "Sat", "sunday": "Sun",
}


@dataclass
class IntentResult:
    fire_think: bool = False
    is_interrupt: bool = False
    is_new_info: bool = False
    constraints_delta: Dict[str, str] = field(default_factory=dict)   # machine: origin=PEK
    human: Dict[str, str] = field(default_factory=dict)               # display: origin=北京
    reason: str = ""

    def __bool__(self) -> bool:  # truthy if anything actionable
        return self.fire_think or (self.is_interrupt and self.is_new_info)


def _contains(text: str, needles) -> Optional[str]:
    for n in needles:
        if n in text:
            return n
    return None


def _extract_constraints(text: str, low: str):
    """Pull origin/dest/date out of an utterance. Returns (delta_codes, human_names)."""
    delta: Dict[str, str] = {}
    human: Dict[str, str] = {}

    found = []  # (pos, code, name, surface)
    for surface, (code, name) in _CITIES.items():
        idx = low.find(surface) if surface.isascii() else text.find(surface)
        if idx >= 0:
            found.append((idx, code, name, surface))
    found.sort()

    has_depart = _contains(text, _DEPART) or _contains(low, _DEPART)
    has_dest = _contains(text, _DEST) or _contains(low, _DEST)

    if found:
        # heuristic: with a departure marker, the first city is the origin
        if has_depart or not has_dest:
            _, code, name, _ = found[0]
            delta["origin"] = code
            human["origin"] = name
            if len(found) > 1 and has_dest:
                _, c2, n2, _ = found[1]
                delta["dest"] = c2
                human["dest"] = n2
        elif has_dest:
            _, code, name, _ = found[0]
            delta["dest"] = code
            human["dest"] = name

    for k, v in _WEEKDAYS.items():
        if (k in low if k.isascii() else k in text):
            delta["date"] = v
            human["date"] = k
            break
    return delta, human


class IntentClassifier:
    def classify(self, transcript: str, task: Optional[TaskState] = None,
                 model_speaking: bool = False) -> IntentResult:
        text = transcript.strip()
        low = text.lower()
        r = IntentResult()

        r.is_interrupt = _contains(text, _INTERRUPT) is not None or _contains(low, _INTERRUPT) is not None
        delta, human = _extract_constraints(text, low)
        r.constraints_delta = delta
        r.human = human
        r.is_new_info = bool(delta)

        knowledge = _contains(text, _KNOWLEDGE) is not None or _contains(low, _KNOWLEDGE) is not None
        # fire a fresh think on a knowledge intent, OR when new info refines an existing task
        existing = bool(task and task.constraints)
        r.fire_think = knowledge or (r.is_new_info and (existing or r.is_interrupt))

        bits = []
        if knowledge:
            bits.append("knowledge")
        if r.is_interrupt:
            bits.append("interrupt")
        if r.is_new_info:
            bits.append("new_info:" + ",".join(f"{k}={v}" for k, v in delta.items()))
        r.reason = " ".join(bits) or "chit-chat"
        return r
