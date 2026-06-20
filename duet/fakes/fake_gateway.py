"""FakeGateway — emulates the MiniCPM duplex gateway's WRITE seam over WebSocket.

It accepts a synthetic user turn (`{"type":"user_text","text":...}`) and replies as
if the model spoke it back (`{"type":"assistant_speak","text":...}`). This lets the
H0 spike script and the write-seam test exercise the real `websockets` plumbing and
message contract before anyone has the real gateway in hand.

This is ONLY a stand-in. The real gateway protocol is what the H0 spike must
discover; when it does, override GatewayAdapter in duet/write_seam.py to match.
"""
from __future__ import annotations

import json
from typing import List, Optional

import websockets


class FakeGateway:
    def __init__(self) -> None:
        self.received: List[str] = []     # texts injected by the WRITE seam
        self._server: Optional[websockets.WebSocketServer] = None
        self.url: Optional[str] = None

    async def _handler(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "user_text":
                text = msg.get("text", "")
                self.received.append(text)
                # the model "speaks" the injected content in its own voice
                await ws.send(json.dumps(
                    {"type": "assistant_speak", "text": f"（模型说）{text}"},
                    ensure_ascii=False))

    async def start(self) -> str:
        self._server = await websockets.serve(self._handler, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"
        return self.url

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()
