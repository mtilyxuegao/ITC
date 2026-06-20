"""Thinker 可调用的工具。目前:web_search(DuckDuckGo,无需 API key)。

给大模型联网能力 —— 否则它只能凭记忆瞎编(如把英伟达股价编成 $549.32)。
"""
from __future__ import annotations

import html
import logging
import re

import aiohttp

logger = logging.getLogger("tt.tools")

# OpenAI/SGLang tool schema
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "联网搜索实时/事实信息(股价、新闻、最新数据、日期、人物等)。返回网页结果摘要。涉及'当前/最新/今天/现在'的事实问题必须先调用它,不要凭记忆回答。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
    },
}

_SNIPPET = re.compile(r"result-snippet['\"]?\s*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return html.unescape(_TAGS.sub("", s)).strip()


async def web_search(query: str, max_results: int = 4, timeout_s: float = 12.0) -> str:
    """用 DuckDuckGo lite 搜索,返回前若干条摘要的纯文本。失败返回提示串(不抛异常)。"""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as r:
                doc = await r.text()
    except Exception as e:  # noqa: BLE001
        logger.warning("web_search failed: %s", e)
        return f"(搜索失败: {e})"

    snippets = [_clean(m) for m in _SNIPPET.findall(doc)]
    snippets = [s for s in snippets if s][:max_results]
    if not snippets:
        return "(未找到相关结果)"
    out = "\n".join(f"- {s}" for s in snippets)
    logger.info("web_search(%r) -> %d results", query, len(snippets))
    return out
