"""Thinker 客户端:把对话上下文丢给 SGLang 上的 Qwen3.5-35B-A3B,拿回一条指令。

取消(abort)做两层(对应 docs 里的"源头取消"):
  1. 取消正在跑的 aiohttp 请求 → 关闭连接。SGLang 检测到客户端断开会**中止该生成**,立刻还卡。
  2. best-effort 调 SGLang 的 /abort_request(若我们带了 rid)。
配合 orchestrator 的 epoch 围栏(汇入作废),即使 abort 晚一步,过期指令也会被丢弃。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import aiohttp

from .config import Config
from .directives import Directive, parse_directive
from .prompts import THINKER_SYSTEM_PROMPT, build_thinker_user_prompt

logger = logging.getLogger("tt.thinker")


class SGLangThinker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None
        self._inflight: Optional[asyncio.Task] = None
        self._inflight_rid: Optional[str] = None

    async def __aenter__(self) -> "SGLangThinker":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.abort()
        if self._session:
            await self._session.close()

    async def think(self, context: str) -> Directive:
        """同步语义:跑一次 Thinker,返回 Directive。可被 abort() 取消(抛 CancelledError)。"""
        assert self._session is not None, "use 'async with SGLangThinker(cfg)'"
        rid = f"tt-{uuid.uuid4().hex[:16]}"
        self._inflight_rid = rid
        payload = {
            "model": self.cfg.thinker_model,
            "messages": [
                {"role": "system", "content": THINKER_SYSTEM_PROMPT},
                {"role": "user", "content": build_thinker_user_prompt(context)},
            ],
            "temperature": self.cfg.thinker_temperature,
            "max_tokens": self.cfg.thinker_max_tokens,
            "rid": rid,  # SGLang 透传,便于 /abort_request 精确取消
            # 关掉思考链:directive 是每 tick 的快速控制决策,长 CoT 既慢又会吃光
            # max_tokens 导致 content 为空(实测 Qwen3.5)。深推理能力靠 35B 本身,不靠显式 CoT。
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Authorization": f"Bearer {self.cfg.thinker_api_key}"}
        url = self.cfg.thinker_base_url.rstrip("/") + "/chat/completions"

        async def _post() -> Directive:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            d = parse_directive(content)
            logger.info("thinker -> %s conf=%.2f reason=%s", d.action.value, d.confidence, d.reason[:60])
            return d

        self._inflight = asyncio.ensure_future(_post())
        try:
            return await self._inflight
        finally:
            self._inflight = None
            self._inflight_rid = None

    async def abort(self) -> None:
        """实时取消正在跑的 Thinker 请求。"""
        task, rid = self._inflight, self._inflight_rid
        if task and not task.done():
            task.cancel()  # 断开连接 → SGLang 中止生成
        # best-effort 显式 abort
        if rid and self._session and self.cfg.thinker_abort_url:
            try:
                async with self._session.post(self.cfg.thinker_abort_url, json={"rid": rid}) as r:
                    await r.read()
            except Exception as e:  # noqa: BLE001 — abort 失败不致命,有 epoch 围栏兜底
                logger.debug("explicit abort failed (ok, fenced anyway): %s", e)
