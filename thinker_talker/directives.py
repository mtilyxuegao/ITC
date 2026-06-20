"""Thinker 控制指令:NOOP / INJECT / CUT,以及从模型输出解析它们。

Thinker 被要求输出一段 JSON(见 prompts.py):
    {"action": "NOOP|INJECT|CUT", "text": "...", "reason": "...", "confidence": 0.0~1.0}

为鲁棒起见,parse_directive 同时支持:
    1. 纯 JSON
    2. 包在 ```json ... ``` 里的 JSON
    3. 文本中第一个 {...} 片段
    4. 行式回退:以 CUT:/INJECT:/NOOP 开头的行
解析失败一律降级为 NOOP(安全:不打断、不乱说)。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Action(str, Enum):
    NOOP = "NOOP"
    INJECT = "INJECT"   # 找间隙补一句更深/更准的话
    CUT = "CUT"         # 立即打断用户并说 text


@dataclass
class Directive:
    action: Action
    text: str = ""              # INJECT/CUT 要说的话(redirect)
    reason: str = ""            # 给日志/调试
    confidence: float = 0.0     # 0~1,CUT 用它做门控

    @property
    def is_noop(self) -> bool:
        return self.action is Action.NOOP

    @staticmethod
    def noop(reason: str = "") -> "Directive":
        return Directive(Action.NOOP, reason=reason)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_action(raw: object) -> Action:
    s = str(raw or "").strip().upper()
    for a in Action:
        if s == a.value:
            return a
    return Action.NOOP


def _coerce_conf(raw: object) -> float:
    try:
        c = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, c))


def parse_directive(model_output: str) -> Directive:
    """把 Thinker 的原始文本输出解析成一个 Directive。永不抛异常。"""
    if not model_output or not model_output.strip():
        return Directive.noop("empty output")

    text = model_output.strip()

    # 1) 去掉 ```json fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None

    # 2) 否则取第一个 {...}
    if candidate is None:
        m = _JSON_BLOCK.search(text)
        candidate = m.group(0) if m else None

    if candidate is not None:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return Directive(
                    action=_coerce_action(obj.get("action")),
                    text=str(obj.get("text") or "").strip(),
                    reason=str(obj.get("reason") or "").strip(),
                    confidence=_coerce_conf(obj.get("confidence")),
                )
        except json.JSONDecodeError:
            pass

    # 3) 行式回退
    head = text.splitlines()[0].strip().upper()
    if head.startswith("CUT"):
        return Directive(Action.CUT, text=text.split(":", 1)[-1].strip(), confidence=1.0, reason="line-fallback")
    if head.startswith("INJECT"):
        return Directive(Action.INJECT, text=text.split(":", 1)[-1].strip(), confidence=1.0, reason="line-fallback")

    return Directive.noop("unparseable -> noop")
