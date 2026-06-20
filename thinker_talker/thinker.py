"""Thinker 客户端:把对话上下文丢给 SGLang 上的 Qwen3.5-35B-A3B,拿回一条指令。

取消(abort)做两层(对应 docs 里的"源头取消"):
  1. 取消正在跑的 aiohttp 请求 → 关闭连接。SGLang 检测到客户端断开会**中止该生成**,立刻还卡。
  2. best-effort 调 SGLang 的 /abort_request(若我们带了 rid)。
配合 orchestrator 的 epoch 围栏(汇入作废),即使 abort 晚一步,过期指令也会被丢弃。
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Optional

import json

import aiohttp

from .config import Config
from .directives import Directive, parse_directive
from .prompts import THINKER_SYSTEM_PROMPT, build_thinker_user_prompt
from .tools import WEB_SEARCH_TOOL, web_search

logger = logging.getLogger("tt.thinker")

# Qwen3.5 的 XML 工具调用格式(SGLang auto 解析器没转成结构化 tool_calls,故客户端兜底解析):
#   <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>
_XML_FN = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
_XML_PARAM = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _extract_tool_calls(msg: dict):
    """从一条 assistant 消息提取 (name, query) 列表;兼容结构化 tool_calls 与 XML 文本。"""
    calls = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append((fn.get("name"), args.get("query", "")))
    if not calls:
        for name, body in _XML_FN.findall(msg.get("content") or ""):
            params = {k: v.strip() for k, v in _XML_PARAM.findall(body)}
            calls.append((name, params.get("query", "")))
    return calls


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

    async def think(self, context: str, on_search=None) -> Directive:
        """跑一次 Thinker(含 web_search 工具调用循环),返回 Directive。
        on_search(query, results): 可选回调,用于把搜索记进对话 log。
        可被 abort() 取消(抛 CancelledError)。"""
        assert self._session is not None, "use 'async with SGLangThinker(cfg)'"
        self._inflight = asyncio.ensure_future(self._run(context, on_search))
        try:
            return await self._inflight
        finally:
            self._inflight = None
            self._inflight_rid = None

    async def _run(self, context: str, on_search) -> Directive:
        headers = {"Authorization": f"Bearer {self.cfg.thinker_api_key}"}
        url = self.cfg.thinker_base_url.rstrip("/") + "/chat/completions"
        messages = [
            {"role": "system", "content": THINKER_SYSTEM_PROMPT},
            {"role": "user", "content": build_thinker_user_prompt(context)},
        ]

        # 最多 3 轮:允许 web_search 工具调用,最后一轮拿到 JSON 指令
        for _round in range(3):
            rid = f"tt-{uuid.uuid4().hex[:16]}"
            self._inflight_rid = rid
            payload = {
                "model": self.cfg.thinker_model,
                "messages": messages,
                "temperature": self.cfg.thinker_temperature,
                "max_tokens": self.cfg.thinker_max_tokens,
                "rid": rid,
                "tools": [WEB_SEARCH_TOOL],
                "tool_choice": "auto",
                # 思考链关掉:directive 要快;深度靠 35B 本身 + web_search。
                "chat_template_kwargs": {"enable_thinking": False},
            }
            async with self._session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            calls = _extract_tool_calls(msg)

            if calls and _round < 2:
                # 执行搜索,把结果作为普通消息回填(避开 tool_call_id 协议的脆弱性),再问一轮
                messages.append({"role": "assistant", "content": content})
                blocks = []
                for name, query in calls:
                    if name != "web_search" or not query:
                        continue
                    result = await web_search(query)
                    if on_search:
                        try:
                            on_search(query, result)
                        except Exception:
                            pass
                    blocks.append(f"[搜索: {query}]\n{result}")
                messages.append({
                    "role": "user",
                    "content": "web_search 结果如下,请据此**只输出一条 JSON 指令**(不要再调用工具):\n"
                               + "\n\n".join(blocks),
                })
                continue

            d = parse_directive(content)
            logger.info("thinker -> %s conf=%.2f reason=%s", d.action.value, d.confidence, d.reason[:60])
            return d

        logger.warning("thinker: tool loop exhausted, NOOP")
        return Directive.noop("tool loop exhausted")

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
