"""会话状态:epoch 围栏 + floor arbiter + 滚动上下文。

设计要点(见 docs/architecture.html §4):
- epoch 是单调递增的 fencing token。用户打断 / 出现新的实质输入 -> epoch += 1。
  Thinker 的每条指令都带它"计算时的 epoch";落地前校验 directive.epoch == state.epoch,
  过期(更小)一律丢弃。这一套同时守住"过期答案"和"过期 CUT"。
- floor arbiter 决定"谁持麦":Talker 自答 / Thinker INJECT / Thinker CUT,避免两声音相撞。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional


class Floor(str, Enum):
    IDLE = "idle"          # 没人说话(LISTENING)
    TALKER = "talker"      # 小模型在说自己的话
    THINKER = "thinker"    # 正在播 Thinker 的 INJECT/CUT


class TalkerState(str, Enum):
    LISTENING = "listening"
    SPEAKING = "speaking"
    THINKING = "thinking"


@dataclass
class Turn:
    speaker: str           # "user" | "talker"
    text: str
    epoch: int
    ts: float


@dataclass
class SessionState:
    session_id: str = ""
    _epoch: int = 0
    talker_state: TalkerState = TalkerState.LISTENING
    floor: Floor = Floor.IDLE
    user_speaking: bool = False
    transcript: Deque[Turn] = field(default_factory=lambda: deque(maxlen=64))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- epoch 围栏 ----
    @property
    def epoch(self) -> int:
        return self._epoch

    def bump_epoch(self) -> int:
        """推进 fence(打断 / 新输入时调用)。返回新 epoch。"""
        with self._lock:
            self._epoch += 1
            return self._epoch

    def is_current(self, epoch: int) -> bool:
        """directive/result 是否仍然有效(没被新 epoch 作废)。"""
        return epoch == self._epoch

    def snapshot_epoch(self) -> int:
        """提交 Thinker 任务前取一份 epoch 快照。"""
        return self._epoch

    # ---- 上下文 ----
    def add_turn(self, speaker: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.transcript.append(Turn(speaker, text, self._epoch, time.time()))

    def looks_like_self_echo(self, text: str, threshold: float = 0.8, lookback: int = 6) -> bool:
        """ASR 转写文本是否其实是 AI 自己刚说的话(经麦克风回采)。

        与最近 lookback 条 talker 草稿做归一化比对:互为子串,或相似度 >= threshold,
        即判为自回声,应丢弃而不是当成用户输入(否则形成 ASR→Thinker→说话→再采的闭环)。
        """
        import difflib
        nt = "".join((text or "").split())
        if not nt:
            return False
        for turn in list(self.transcript)[-lookback:]:
            if turn.speaker != "talker":
                continue
            ns = "".join(turn.text.split())
            if not ns:
                continue
            if nt in ns or ns in nt:
                return True
            if difflib.SequenceMatcher(None, nt, ns).ratio() >= threshold:
                return True
        return False

    def recent_context(self, max_turns: int = 12) -> List[Turn]:
        items = list(self.transcript)
        return items[-max_turns:]

    def render_context(self, max_turns: int = 12) -> str:
        lines = []
        for t in self.recent_context(max_turns):
            who = "用户" if t.speaker == "user" else "助手(小模型草稿)"
            lines.append(f"{who}: {t.text}")
        return "\n".join(lines)


class FloorArbiter:
    """谁持麦的仲裁。规则:
    - CUT 可抢占 Talker 与 IDLE(这是 CUT 的本职)。
    - INJECT 仅在 IDLE 时落地(礼貌:等小模型说完/停顿)。
    - 用户打断永远能把麦抢回 IDLE(由 orchestrator 在 epoch++ 时调 release)。
    """

    def __init__(self, state: SessionState):
        self.state = state

    def can_inject(self) -> bool:
        return self.state.floor in (Floor.IDLE,)

    def can_cut(self) -> bool:
        return self.state.floor in (Floor.IDLE, Floor.TALKER)

    def take(self, who: Floor) -> None:
        self.state.floor = who

    def release(self) -> None:
        self.state.floor = Floor.IDLE
