"""FakeVLLM — an OpenAI-compatible streaming chat server with tool calls.

Deterministic and origin-aware so the flight demo always produces the same answer
(and a DIFFERENT answer for PVG vs PEK, which lets the scenario test prove that the
re-dispatched epoch used the corrected origin). `chunk_delay` spaces out the SSE
chunks so a test can interrupt mid-stream.
"""
from __future__ import annotations

import asyncio
import json
import socket
from typing import List, Optional

from aiohttp import web


def _chunk(delta: dict, finish: Optional[str] = None) -> bytes:
    obj = {"id": "fakecmpl", "object": "chat.completion.chunk",
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


class FakeVLLM:
    def __init__(self, chunk_delay: float = 0.0) -> None:
        self.chunk_delay = chunk_delay
        self.requests: List[dict] = []      # every payload received (for assertions)
        self._runner: Optional[web.AppRunner] = None
        self._sock: Optional[socket.socket] = None
        self.base_url: Optional[str] = None

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        port = self._sock.getsockname()[1]
        site = web.SockSite(self._runner, self._sock)
        await site.start()
        self.base_url = f"http://127.0.0.1:{port}"
        return self.base_url

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    async def _send(self, resp: web.StreamResponse, payload: bytes) -> None:
        if self.chunk_delay:
            await asyncio.sleep(self.chunk_delay)
        await resp.write(payload)

    async def _handler(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        messages = body.get("messages", [])
        has_tools = bool(body.get("tools"))
        joined = " ".join(m.get("content") or "" for m in messages
                          if isinstance(m.get("content"), str))
        origin = "PEK" if ("北京" in joined or "PEK" in joined) else "PVG"
        tool_msgs = [m for m in messages if m.get("role") == "tool"]

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                           "Cache-Control": "no-cache"})
        await resp.prepare(request)

        try:
            if has_tools and not tool_msgs:
                # round 1: think out loud, then call the flight tool
                await self._send(resp, _chunk({"role": "assistant"}))
                await self._send(resp, _chunk({"content": "让我查一下"}))
                await self._send(resp, _chunk({"tool_calls": [
                    {"index": 0, "id": "call_0", "type": "function",
                     "function": {"name": "flight_search", "arguments": ""}}]}))
                args = json.dumps({"origin": origin, "date": "Fri"})
                await self._send(resp, _chunk({"tool_calls": [
                    {"index": 0, "function": {"arguments": args}}]}))
                await self._send(resp, _chunk({}, finish="tool_calls"))
            else:
                # round 2: compose the final answer from the tool result
                cheapest = {"code": "ZZ001", "price": 999}
                for m in tool_msgs:
                    try:
                        data = json.loads(m.get("content") or "{}")
                        if "cheapest" in data:
                            cheapest = data["cheapest"]
                            origin = data.get("origin", origin)
                    except json.JSONDecodeError:
                        pass
                answer = f"最便宜的是 {cheapest['code']}，¥{cheapest['price']}（{origin} 出发）"
                await self._send(resp, _chunk({"role": "assistant"}))
                await self._send(resp, _chunk({"content": answer}))
                await self._send(resp, _chunk({}, finish="stop"))

            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            # client dropped the stream mid-generation: this is exactly the [WAIT]
            # abort. A real server frees the KV here; we just stop cleanly.
            pass
        return resp
