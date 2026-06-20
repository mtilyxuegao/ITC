"""Web search tool for the Thinker (THINKER_TALKER.md allowed_tools).

needs_search()  — cheap heuristic: does this question want current/external facts?
web_search()    — runs the search (DuckDuckGo by default, no API key; Tavily/Brave
                  if SEARCH_BACKEND + SEARCH_API_KEY are set) and returns compact
                  results the Thinker can reason over.

The search is run in a thread (ddgs is blocking) so it never stalls the event loop.
Failures degrade gracefully to [] — the Thinker then reasons without external facts.
"""
from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger("thinker-talker.tools")

# Cues that the answer depends on current / external / lookup-able facts.
_NEEDS_SEARCH = re.compile(
    r"\b(latest|current(ly)?|today|todays|tonight|right now|as of|recent(ly)?|"
    r"this (year|week|month)|these days|nowadays|news|update on|"
    r"who (is|are|won|leads)|what is the (price|population|capital|weather|score|status)|"
    r"how much (is|does|are)|when (did|does|will|is)|where is|release date|released|"
    r"stock|weather|forecast|score|standings|election|happening|"
    r"20(2[4-9]|3[0-9]))\b",
    re.IGNORECASE,
)


def needs_search(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(_NEEDS_SEARCH.search(t))


async def web_search(query: str, cfg) -> list[dict]:
    backend = getattr(cfg, "search_backend", "ddgs")
    n = getattr(cfg, "search_max_results", 5)
    try:
        if backend == "tavily" and cfg.search_api_key:
            return await _tavily(query, cfg.search_api_key, n)
        if backend == "brave" and cfg.search_api_key:
            return await _brave(query, cfg.search_api_key, n)
        return await asyncio.get_running_loop().run_in_executor(None, _ddgs, query, n)
    except Exception as e:  # noqa: BLE001
        logger.warning("web_search failed (%s): %s", backend, e)
        return []


def _ddgs(query: str, n: int) -> list[dict]:
    from ddgs import DDGS
    out: list[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=n):
            out.append({
                "title": r.get("title", ""),
                "url": r.get("href", "") or r.get("url", ""),
                "snippet": r.get("body", "") or r.get("snippet", ""),
            })
    return out


async def _tavily(query: str, key: str, n: int) -> list[dict]:
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as hc:
        r = await hc.post("https://api.tavily.com/search", json={
            "api_key": key, "query": query, "max_results": n,
        })
        data = r.json()
    return [{"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": x.get("content", "")} for x in data.get("results", [])]


async def _brave(query: str, key: str, n: int) -> list[dict]:
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as hc:
        r = await hc.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": n},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
        )
        data = r.json()
    return [{"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": x.get("description", "")}
            for x in data.get("web", {}).get("results", [])]


def format_results_for_prompt(query: str, results: list[dict]) -> str:
    if not results:
        return f"(web search for '{query}' returned no usable results)"
    lines = [f"Web search results for '{query}':"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    {r['snippet'][:300]}\n    ({r['url']})")
    return "\n".join(lines)
