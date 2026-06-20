"""Talker 客户端:按 MiniCPM-o demo 的 WS 协议与 gateway 通信。

读取 `response.output.delta`(kind = text / audio / listen),发出:
  - input.append(音频分片;可带 force_listen=true)—— 用于驱动/测试
  - control.force_speak(新增,需 patches/integrate_force_speak.py 注入到 server.py)—— CUT/INJECT 落地

集成说明:生产里浏览器才是媒体客户端,编排层应当**旁路监听**同一会话(需要给 gateway 加一个
observer 钩子)。本类既可作为该 observer 的实现骨架,也可作为独立测试驱动(自己发音频、收草稿)。
协议消息类型见 demos/minicpm-o-4.5-fullduplex 与 core/schemas/duplex.py。
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

import websockets

logger = logging.getLogger("tt.talker")


@dataclass
class TalkerEvent:
    kind: str                      # "text" | "audio" | "listen" | "created" | "closed"
    text: str = ""
    audio_b64: Optional[str] = None
    response_id: Optional[str] = None
    end_of_turn: bool = False
    raw: Optional[dict] = None


class TalkerClient:
    def __init__(self, ws_url: str, mode: str = "full_duplex", system_prompt: Optional[str] = None):
        self.ws_url = ws_url
        self.mode = mode
        self.system_prompt = system_prompt
        self._ws = None
        self._send_lock = asyncio.Lock()

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.ws_url, max_size=None)
        init = {"type": "session.init", "payload": {"mode": self.mode}}
        if self.system_prompt:
            init["payload"]["system_prompt"] = self.system_prompt
        await self._ws.send(json.dumps(init))
        logger.info("talker connected: %s", self.ws_url)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def send_audio(self, audio_b64: str, force_listen: bool = False, input_id: Optional[str] = None) -> None:
        """驱动/测试用:推一段 200ms~1s 的音频分片。"""
        msg = {
            "type": "input.append",
            "payload": {"audio_base64": audio_b64, "hints": {"force_listen": force_listen}},
        }
        if input_id:
            msg["payload"]["input_id"] = input_id
        await self._send(msg)

    async def force_speak(self, text: str, input_id: Optional[str] = None) -> None:
        """让 Talker 立刻打断并说出 text(走注入的 control.force_speak)。"""
        msg = {"type": "control.force_speak", "payload": {"text": text}}
        if input_id:
            msg["payload"]["input_id"] = input_id
        await self._send(msg)
        logger.info("force_speak -> %r", text[:60])

    async def _send(self, msg: dict) -> None:
        if self._ws is None:
            raise RuntimeError("talker not connected")
        async with self._send_lock:
            await self._ws.send(json.dumps(msg))

    async def events(self):
        """异步迭代 Talker 的输出事件。"""
        if self._ws is None:
            raise RuntimeError("talker not connected")
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type")
            if mtype == "session.created":
                yield TalkerEvent("created", raw=msg)
            elif mtype == "session.closed":
                yield TalkerEvent("closed", raw=msg)
                break
            elif mtype == "response.output.delta":
                kind = msg.get("kind")
                yield TalkerEvent(
                    kind=kind or "unknown",
                    text=msg.get("text", "") or "",
                    audio_b64=msg.get("audio"),
                    response_id=msg.get("response_id"),
                    end_of_turn=(kind == "listen"),
                    raw=msg,
                )
