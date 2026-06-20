"""Thinker 的 OpenAI 实现:GPT 5.4 + Responses API 内置 web_search(真·联网)。

与 SGLangThinker 接口一致(think/abort/async-context),orchestrator 可直接替换。
相比自建 SGLang + DDG,这里的 web search 由 OpenAI 托管,能拿到精确实时数据(如股价)。
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import aiohttp

from .config import Config
from .directives import Directive, parse_directive
from .prompts import THINKER_SYSTEM_PROMPT, build_thinker_user_prompt

logger = logging.getLogger("tt.gpt")


def _extract_text_and_searches(resp: dict):
    """从 Responses API 输出里取最终文本 + web_search 查询列表。"""
    text_parts, searches = [], []
    for item in resp.get("output", []) or []:
        itype = item.get("type")
        if itype == "web_search_call":
            action = item.get("action") or {}
            q = action.get("query") or " ".join(action.get("queries") or [])
            if q:
                searches.append(q)
        elif itype == "message":
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text" and c.get("text"):
                    text_parts.append(c["text"])
    # 兜底:有些返回带顶层 output_text
    if not text_parts and resp.get("output_text"):
        text_parts.append(resp["output_text"])
    return "".join(text_parts), searches


class GPTThinker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.openai_api_key or os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""
        self.model = cfg.openai_model
        self.url = cfg.openai_base_url.rstrip("/") + "/responses"
        self._session: Optional[aiohttp.ClientSession] = None
        self._inflight: Optional[asyncio.Task] = None
        self.last_latency_ms = 0
        if not self.api_key:
            raise RuntimeError("缺 OpenAI key:在 .env 设 OPENAI_TOKEN")

    async def __aenter__(self) -> "GPTThinker":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.abort()
        if self._session:
            await self._session.close()

    async def think(self, context: str, on_search=None) -> Directive:
        assert self._session is not None, "use 'async with GPTThinker(cfg)'"
        self._inflight = asyncio.ensure_future(self._run(context, on_search))
        try:
            return await self._inflight
        finally:
            self._inflight = None

    async def _run(self, context: str, on_search) -> Directive:
        body = {
            "model": self.model,
            "instructions": THINKER_SYSTEM_PROMPT,
            "input": build_thinker_user_prompt(context),
            "tools": [{"type": "web_search"}],
            # 低推理强度:directive 是每 tick 的快决策,够用且更快
            "reasoning": {"effort": "low"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        logger.info("gpt INPUT(发给大模型,证明已传入): %r", body["input"][:300])
        t0 = time.time()
        async with self._session.post(self.url, json=body, headers=headers) as resp:
            if resp.status >= 400:
                detail = (await resp.text())[:300]
                logger.warning("openai error %s: %s", resp.status, detail)
                return Directive.noop(f"openai {resp.status}")
            data = await resp.json()
        self.last_latency_ms = int((time.time() - t0) * 1000)

        text, searches = _extract_text_and_searches(data)
        for q in searches:
            if on_search:
                try:
                    on_search(q, "(OpenAI web_search)")
                except Exception:
                    pass
        d = parse_directive(text)
        logger.info("gpt-thinker -> %s conf=%.2f searches=%d reason=%s",
                    d.action.value, d.confidence, len(searches), d.reason[:50])
        return d

    async def abort(self) -> None:
        task = self._inflight
        if task and not task.done():
            task.cancel()  # 断开连接;epoch 围栏兜底作废过期结果
