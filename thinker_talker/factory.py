"""按 cfg.thinker_provider 选择 Thinker 实现。"""
from __future__ import annotations

from .config import Config


def make_thinker(cfg: Config):
    """openai -> GPTThinker(GPT 5.4 + web_search);sglang -> SGLangThinker(自建)。"""
    if cfg.thinker_provider == "openai":
        from .gpt_thinker import GPTThinker
        return GPTThinker(cfg)
    from .thinker import SGLangThinker
    return SGLangThinker(cfg)
