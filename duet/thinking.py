"""Thinking layer client — the async, cancellable bridge to the MoE + agents.

One shared OpenAI-compatible server (Qwen3.5-35B-A3B on vLLM/SGLang) serves ALL
background agents via continuous batching. This client:
  - streams a chat completion,
  - lets the model call local agent tools (web_search / flight_search / ...),
  - emits MILESTONE updates as work progresses (the user hears "I'm checking..."),
  - emits a final RESULT,
  - is fully CANCELLABLE: cancelling the asyncio task closes the HTTP stream
    (which on real vLLM/SGLang is the abort that frees KV — see duet/README.md).

Correctness of [WAIT] does NOT depend on the abort landing instantly; the
Conductor's epoch guard drops anything stale. This client also self-checks
`is_current()` as defense in depth so a stale epoch stops doing work early.
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, Dict, List, Optional

import aiohttp

MILESTONE = "milestone"
RESULT = "result"

EmitFn = Callable[[str, int, str], Awaitable[None]]
IsCurrentFn = Callable[[], bool]


# ----------------------------- demo agent tools -----------------------------
# Deterministic mocks so the travel demo always lands. Real agents (web/code)
# slot in behind the same name->callable contract.

def flight_search(origin: str = "PVG", date: str = "Fri", **_) -> dict:
    table = {
        "PVG": [{"code": "MU208", "price": 612, "dep": "08:05"},
                {"code": "FM134", "price": 690, "dep": "13:40"}],
        "PEK": [{"code": "CA1234", "price": 588, "dep": "09:10"},
                {"code": "HU742", "price": 640, "dep": "07:20"}],
    }
    flights = table.get(origin, [{"code": "ZZ001", "price": 999, "dep": "12:00"}])
    cheapest = min(flights, key=lambda f: f["price"])
    return {"origin": origin, "date": date, "count": len(flights),
            "flights": flights, "cheapest": cheapest}


def web_search(query: str = "", **_) -> dict:
    return {"query": query, "count": 2,
            "results": [{"title": "结果A", "snippet": "…"},
                        {"title": "结果B", "snippet": "…"}]}


DEFAULT_TOOLS: Dict[str, Callable[..., dict]] = {
    "flight_search": flight_search,
    "web_search": web_search,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "flight_search",
        "description": "Search flights departing from an airport; returns the cheapest option.",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string"}, "date": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Web search.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}}}},
]


class ThinkingClient:
    def __init__(self, base_url: str, model: str = "thinker",
                 tools: Optional[Dict[str, Callable[..., dict]]] = None,
                 tool_schemas=None, request_timeout: float = 60.0,
                 use_tools: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tools = tools if tools is not None else DEFAULT_TOOLS
        self.tool_schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS
        self.request_timeout = request_timeout
        # When the served model/endpoint has no tool-choice enabled, set False to
        # run a pure streaming chat (validates the same stream/abort code path).
        self.use_tools = use_tools
        self.aborts = 0  # number of runs cut short by cancellation (test/telemetry)

    async def _post_stream(self, session, payload):
        url = self.base_url + "/v1/chat/completions"
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.request_timeout)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line == "[DONE]":
                    return
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def _chat(self, session, messages, use_tools: bool):
        payload = {"model": self.model, "messages": messages, "stream": True}
        if use_tools:
            payload["tools"] = self.tool_schemas
            payload["tool_choice"] = "auto"
        content: List[str] = []
        calls: Dict[int, dict] = {}
        finish = None
        async for chunk in self._post_stream(session, payload):
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for tcd in (delta.get("tool_calls") or []):
                slot = calls.setdefault(tcd.get("index", 0),
                                        {"id": None, "name": None, "args": ""})
                if tcd.get("id"):
                    slot["id"] = tcd["id"]
                fn = tcd.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
        tool_calls = [calls[i] for i in sorted(calls)]
        return "".join(content), tool_calls, finish

    async def run(self, epoch: int, task, emit: EmitFn,
                  is_current: Optional[IsCurrentFn] = None) -> None:
        """Drive one thinking episode for `epoch`. Cancellation-safe."""
        is_current = is_current or (lambda: True)
        origin = task.constraints.get("origin", "PVG")
        date = task.constraints.get("date", "Fri")
        try:
            await emit(MILESTONE, epoch, f"正在查询 {origin} 出发的航班…")
            async with aiohttp.ClientSession() as session:
                messages = [
                    {"role": "system", "content": "你是出行助手，必要时调用工具查询，然后用一句话给出结论。"},
                    {"role": "user", "content": f"帮我订 {date} 从 {origin} 出发最便宜的航班。"},
                ]
                content, tool_calls, finish = await self._chat(session, messages, use_tools=True)

                if tool_calls:
                    messages.append({"role": "assistant", "content": content or None,
                                     "tool_calls": [
                                         {"id": c["id"] or f"call_{i}", "type": "function",
                                          "function": {"name": c["name"],
                                                       "arguments": c["args"] or "{}"}}
                                         for i, c in enumerate(tool_calls)]})
                    for i, c in enumerate(tool_calls):
                        if not is_current():
                            return
                        name = c["name"]
                        try:
                            args = json.loads(c["args"] or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        fn = self.tools.get(name)
                        result = fn(**args) if fn else {"error": f"unknown tool {name}"}
                        await emit(MILESTONE, epoch,
                                   f"{name} 找到 {result.get('count', '?')} 个结果")
                        messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                                         "name": name, "content": json.dumps(result, ensure_ascii=False)})
                    if not is_current():
                        return
                    content, _, _ = await self._chat(session, messages, use_tools=False)

            if not is_current():
                return
            await emit(RESULT, epoch, content or "（没有可用结果）")
        except asyncio.CancelledError:
            self.aborts += 1
            # stream is closed by the `async with` on the way out; this is our
            # [WAIT] abort. Re-raise so the awaiting canceller sees clean cancellation.
            raise
