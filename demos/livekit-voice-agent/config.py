"""Central typed config for the Thinker-Talker layer.

Loaded once from environment (.env via dotenv). Kept dependency-light (no
livekit / openai imports) so the other modules stay unit-testable without a
live session. See docs/THINKER_TALKER.md for the design this configures.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _envs(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _envb(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# Default filler lines spoken while the Thinker works (latency hiding, §3.4/§5).
_DEFAULT_FILLERS = [
    "Let me think about that for a second.",
    "Good question, give me a moment.",
    "Let me work that out.",
]


@dataclass
class ThinkerConfig:
    # --- Thinker (large reasoning model on vLLM/SGLang, OpenAI-compatible) ---
    base_url: str = "http://localhost:8000/v1"
    model: str = "itc-thinker"            # served-model-name passed to vllm serve
    api_key: str = "EMPTY"                # vLLM ignores it; must be non-empty for the SDK
    backend: str = "vllm"                 # "vllm" | "sglang" -> selects abort strategy
    # QwQ-32B is a verbose reasoner (long CoT before the answer). 8s/1024tok is the
    # doc's suggested production cap, but QwQ measured ~35s on a trivial question and
    # ~55 tok/s; tune via env. For a snappier demo, swap to a faster Thinker model.
    timeout_s: float = 25.0               # §7 hard cap on a single thought
    max_tokens: int = 2048                # reasoning + answer (reasoning split out by vLLM)
    temperature: float = 0.6

    # --- Escalation / behavior ---
    escalation_enabled: bool = True       # kill switch -> pure Talker behavior
    # WHO decides to wake the Thinker:
    #   "talker" -> the Talker LLM calls the think_deeply tool when it judges it needed (default)
    #   "auto"   -> heuristic keyword classifier auto-escalates (the old behavior)
    #   "off"    -> never escalate
    escalation_mode: str = "talker"
    min_words_for_length_trigger: int = 12
    fillers: list[str] = field(default_factory=lambda: list(_DEFAULT_FILLERS))

    # --- Web search tool (Thinker) ---
    search_enabled: bool = True
    search_backend: str = "ddgs"          # ddgs (free) | tavily | brave | none
    search_api_key: str = ""              # for tavily/brave
    search_max_results: int = 5

    # --- Visualization dashboard (fire-and-forget event sink) ---
    dashboard_url: str = "http://localhost:8800"

    # Fallback line spoken if the Thinker times out / errors (§7).
    fallback_text: str = "I'm not totally sure on that one, but here's my best take."


def load_config() -> ThinkerConfig:
    return ThinkerConfig(
        base_url=_envs("THINKER_BASE_URL", "http://localhost:8000/v1"),
        model=_envs("THINKER_MODEL", "itc-thinker"),
        api_key=_envs("THINKER_API_KEY", "EMPTY"),
        backend=_envs("THINKER_BACKEND", "vllm").lower(),
        timeout_s=_envf("THINKER_TIMEOUT_S", 25.0),
        max_tokens=_envi("THINKER_MAX_TOKENS", 2048),
        temperature=_envf("THINKER_TEMPERATURE", 0.6),
        escalation_enabled=_envb("ESCALATION_ENABLED", True),
        escalation_mode=_envs("ESCALATION_MODE", "talker").lower(),
        min_words_for_length_trigger=_envi("ESCALATION_MIN_WORDS", 12),
        search_enabled=_envb("SEARCH_ENABLED", True),
        search_backend=_envs("SEARCH_BACKEND", "ddgs").lower(),
        search_api_key=_envs("SEARCH_API_KEY", ""),
        search_max_results=_envi("SEARCH_MAX_RESULTS", 5),
        dashboard_url=_envs("DASHBOARD_URL", "http://localhost:8800"),
    )
