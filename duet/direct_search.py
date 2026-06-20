"""DirectSearchClient — the fast research path that REPLACES the hermes agent loop.

Measured on this cluster: ddgs web_search ~1-2s DOMINATES; an LLM summarize is ~120ms. The
gemma router already turned the user's turn into a standalone query, so hermes's decide-to-search
LLM hop + agent-framework overhead + extra fetches were pure latency. This does only:
    ddgs.text(query)  ->  ONE LLM summarize  ->  one short spoken sentence
cutting the research window from ~3-6s to ~1.5-2.2s.

Same run(epoch, task, emit, is_current) contract + MILESTONE/RESULT + cancellable as the hermes
client, so the Conductor, the progress cue (fires on the first MILESTONE), and the overlay are
unchanged. Falls back gracefully (empty snippets / errors -> a spoken "couldn't find it").
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import os

import aiohttp

from .thinking import MILESTONE, RESULT, EmitFn


class DirectSearchClient:
    def __init__(self, llm_url: str, model: str = "qwen", n_results: int = 5) -> None:
        self.llm_url = llm_url if llm_url.rstrip("/").endswith("/v1") else llm_url.rstrip("/") + "/v1"
        self.model = model
        self.n_results = n_results
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.aborts = 0

    def _search(self, query: str) -> str:
        """Blocking ddgs web search -> compact snippet block. ddgs spams stderr trying to parse
        the cluster /etc/hosts; redirect it so the proxy log stays clean."""
        try:
            from ddgs import DDGS
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                rs = list(DDGS().text(query, max_results=self.n_results))
        except Exception as e:
            print(f"[duet] ddgs search error: {e}", flush=True)
            return ""
        return "\n".join(f"{i + 1}. {r.get('title', '')}: {r.get('body', '')}"[:300]
                         for i, r in enumerate(rs))

    async def _summarize(self, query: str, snippets: str) -> str:
        sys_p = ("Answer the user's question in ONE short natural spoken sentence (under 18 words) "
                 "from the search snippets. Spell numbers as plain rounded words (e.g. 'about two "
                 "hundred ten dollars'); no symbols, no $, no decimals, no ranges, no URLs, no "
                 "markdown. If the snippets do not contain the answer, say you could not find it.")
        payload = {"model": self.model, "stream": False, "max_tokens": 64, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "system", "content": sys_p},
                                {"role": "user",
                                 "content": f"Question: {query}\nSnippets:\n{snippets}"}]}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(self.llm_url + "/chat/completions", json=payload,
                                  timeout=aiohttp.ClientTimeout(total=20)) as r:
                    d = await r.json()
            return (d["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            print(f"[duet] summarize error: {e}", flush=True)
            return ""

    async def run(self, epoch: int, task, emit: EmitFn, is_current=None) -> None:
        query = (task.dialog[-1] if getattr(task, "dialog", None) else task.summary()).strip()
        await emit(MILESTONE, epoch, f"web_search: {query}")
        loop = asyncio.get_running_loop()
        snippets = await loop.run_in_executor(self._executor, self._search, query)
        if is_current and not is_current():
            return
        await emit(MILESTONE, epoch, "summarizing results")
        answer = await self._summarize(query, snippets)
        if is_current and not is_current():
            return
        await emit(RESULT, epoch, answer or "I could not find that just now.")
