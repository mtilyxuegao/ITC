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

    # Thinker 后端选择: openai(GPT + 内置 web search) | sglang(自建)
    thinker_provider: str = "openai"

    # Thinker — SGLang(自建)
    thinker_base_url: str = "http://localhost:30000/v1"
    thinker_model: str = "Qwen/Qwen3.5-35B-A3B"
    thinker_api_key: str = "EMPTY"
    thinker_abort_url: str = "http://localhost:30000/abort_request"
    thinker_temperature: float = 0.6
    thinker_max_tokens: int = 512

    # Thinker — OpenAI(GPT 5.4 + Responses API 内置 web_search)
    openai_model: str = "gpt-5.4"
    openai_base_url: str = "https://api.openai.com/v1"
    # key 从环境 OPENAI_TOKEN(或 OPENAI_API_KEY)读
    openai_api_key: str = ""

    # 策略
    thinker_tick_seconds: float = 1.0
    cut_min_confidence: float = 0.7
    interrupt_priority: str = "human_wins"  # human_wins | ai_can_override
    inject_wait_for_gap: bool = True

    # ASR: openai(转写API) | local(本地 faster-whisper,OpenAI 兼容接口)
    asr_provider: str = "openai"
    asr_base_url: str = "https://api.openai.com/v1"
    asr_model: str = "gpt-4o-mini-transcribe"
    asr_language: str = "zh"
    asr_api_key: str = ""  # local 不需要
    # 防自听闭环:AI 持麦/刚说完的尾窗内丢弃上行音频(秒);ASR 文本与近期草稿的相似度阈值
    asr_echo_guard_seconds: float = 0.6
    asr_echo_sim_threshold: float = 0.8

    # 对话 log:记录小模型草稿 + 大模型决策/搜索/插话(用于核对大模型是否真被调用)
    transcript_log_path: str = ""

    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            talker_gateway_ws=os.environ.get("TALKER_GATEWAY_WS", cls.talker_gateway_ws),
            talker_session_mode=os.environ.get("TALKER_SESSION_MODE", cls.talker_session_mode),
            thinker_provider=os.environ.get("THINKER_PROVIDER", cls.thinker_provider),
            openai_model=os.environ.get("OPENAI_MODEL", cls.openai_model),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", cls.openai_base_url),
            openai_api_key=os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY") or "",
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
            asr_echo_guard_seconds=_f("ASR_ECHO_GUARD_SECONDS", cls.asr_echo_guard_seconds),
            asr_echo_sim_threshold=_f("ASR_ECHO_SIM_THRESHOLD", cls.asr_echo_sim_threshold),
            transcript_log_path=os.environ.get("TRANSCRIPT_LOG_PATH", cls.transcript_log_path),
            log_level=os.environ.get("LOG_LEVEL", cls.log_level),
        )
