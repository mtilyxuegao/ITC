"""编排层配置:全部从环境变量读取(见 .env.thinker-talker.example)。"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # Talker
    talker_gateway_ws: str = "ws://localhost:8006/ws"
    talker_session_mode: str = "full_duplex"

    # Thinker
    thinker_base_url: str = "http://localhost:30000/v1"
    thinker_model: str = "Qwen/Qwen3.5-35B-A3B"
    thinker_api_key: str = "EMPTY"
    thinker_abort_url: str = "http://localhost:30000/abort_request"
    thinker_temperature: float = 0.6
    thinker_max_tokens: int = 512

    # 策略
    thinker_tick_seconds: float = 1.0
    cut_min_confidence: float = 0.7
    interrupt_priority: str = "human_wins"  # human_wins | ai_can_override
    inject_wait_for_gap: bool = True

    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            talker_gateway_ws=os.environ.get("TALKER_GATEWAY_WS", cls.talker_gateway_ws),
            talker_session_mode=os.environ.get("TALKER_SESSION_MODE", cls.talker_session_mode),
            thinker_base_url=os.environ.get("THINKER_BASE_URL", cls.thinker_base_url),
            thinker_model=os.environ.get("THINKER_MODEL", cls.thinker_model),
            thinker_api_key=os.environ.get("THINKER_API_KEY", cls.thinker_api_key),
            thinker_abort_url=os.environ.get("THINKER_ABORT_URL", cls.thinker_abort_url),
            thinker_temperature=_f("THINKER_TEMPERATURE", cls.thinker_temperature),
            thinker_max_tokens=int(_f("THINKER_MAX_TOKENS", cls.thinker_max_tokens)),
            thinker_tick_seconds=_f("THINKER_TICK_SECONDS", cls.thinker_tick_seconds),
            cut_min_confidence=_f("CUT_MIN_CONFIDENCE", cls.cut_min_confidence),
            interrupt_priority=os.environ.get("INTERRUPT_PRIORITY", cls.interrupt_priority),
            inject_wait_for_gap=_b("INJECT_WAIT_FOR_GAP", cls.inject_wait_for_gap),
            log_level=os.environ.get("LOG_LEVEL", cls.log_level),
        )
