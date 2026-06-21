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

    async def force_speak(self, text: str, input_id: Optional[str] = None, interrupt: bool = False) -> None:
        """让 Talker 说出 text(走注入的 control.force_speak)。
        interrupt=True(CUT):先停掉/flush 当前在说的话再说 text;False(INJECT):排队接在后面。"""
        payload = {"text": text, "interrupt": interrupt}
        if input_id:
            payload["input_id"] = input_id
        await self._send({"type": "control.force_speak", "payload": payload})
        logger.info("force_speak(interrupt=%s) -> %r", interrupt, text[:60])

    async def stop(self, input_id: Optional[str] = None) -> None:
        """立即停掉 Talker 当前发声并 flush 残音(走注入的 control.stop),不接新话。"""
        payload = {}
        if input_id:
            payload["input_id"] = input_id
        await self._send({"type": "control.stop", "payload": payload})
        logger.info("stop")

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


class GatewayObserver:
    """旁路监听一个**已存在**的浏览器↔worker 会话(经 gateway 的 /observer 钩子)。

    与 TalkerClient(自己当客户端驱动会话)不同,这个不占用会话,而是:
      - 通过 GET  {base}/observer/sessions 发现活跃会话
      - 连    WS   {base}/observer/{sid} 收 text/listen 镜像
      - 发 control.force_speak → gateway 注入该会话的 worker(实现 AI 主动打断)
    实测路径见 tests/e2e_observer.py。与 Orchestrator 的事件接口一致(events()/force_speak())。
    """

    def __init__(self, base_url: str, session_id: Optional[str] = None, insecure: bool = True):
        # base_url 形如 https://localhost:8006
        self.base = base_url.rstrip("/")
        self.session_id = session_id
        self._ssl = None
        if base_url.startswith("https"):
            import ssl
            self._ssl = ssl._create_unverified_context() if insecure else ssl.create_default_context()
        self._ws = None
        self._send_lock = asyncio.Lock()
        self._watch_task = None

    def _ws_url(self, path: str) -> str:
        return self.base.replace("http", "ws", 1) + path

    async def _watch_stale(self) -> None:
        """看门狗:轮询 /observer/sessions。若当前会话消失(用户刷新/结束)或出现更新的会话,
        关掉当前 ws —— 让 events() 结束、上层重连到最新会话(修复'刷新后 ASR/编排不再工作')。"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as s:
                while self._ws is not None:
                    await asyncio.sleep(3.0)
                    try:
                        async with s.get(self.base + "/observer/sessions", ssl=self._ssl) as r:
                            sessions = (await r.json()).get("sessions") or []
                    except Exception:
                        continue
                    latest = sessions[-1] if sessions else None
                    if (self.session_id not in sessions) or (latest and latest != self.session_id):
                        logger.info("observer: session %s stale (latest=%s) -> reconnect", self.session_id, latest)
                        try:
                            if self._ws is not None:
                                await self._ws.close()
                        except Exception:
                            pass
                        return
        except asyncio.CancelledError:
            pass

    async def discover_session(self, timeout_s: float = 30.0) -> Optional[str]:
        import aiohttp
        deadline = timeout_s
        async with aiohttp.ClientSession() as s:
            while deadline > 0:
                try:
                    async with s.get(self.base + "/observer/sessions", ssl=self._ssl) as r:
                        data = await r.json()
                        sessions = data.get("sessions") or []
                        if sessions:
                            return sessions[-1]
                except Exception:
                    pass
                await asyncio.sleep(1.0)
                deadline -= 1.0
        return None

    async def connect(self) -> None:
        if self.session_id is None:
            self.session_id = await self.discover_session()
        if self.session_id is None:
            raise RuntimeError("no active session to observe")
        self._ws = await websockets.connect(self._ws_url(f"/observer/{self.session_id}"),
                                             ssl=self._ssl, max_size=None)
        logger.info("observing session %s", self.session_id)
        self._watch_task = asyncio.create_task(self._watch_stale())

    async def close(self) -> None:
        if self._watch_task is not None:
            self._watch_task.cancel()
            self._watch_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def force_speak(self, text: str, input_id: Optional[str] = None, interrupt: bool = False) -> None:
        if self._ws is None:
            raise RuntimeError("observer not connected")
        payload = {"text": text, "interrupt": interrupt}
        if input_id:
            payload["input_id"] = input_id
        async with self._send_lock:
            await self._ws.send(json.dumps({"type": "control.force_speak", "payload": payload}))
        logger.info("force_speak(interrupt=%s) -> %r", interrupt, text[:60])

    async def stop(self, input_id: Optional[str] = None) -> None:
        if self._ws is None:
            raise RuntimeError("observer not connected")
        payload = {"input_id": input_id} if input_id else {}
        async with self._send_lock:
            await self._ws.send(json.dumps({"type": "control.stop", "payload": payload}))
        logger.info("stop")

    async def send_status(self, payload: dict) -> None:
        """把编排层状态/决策推给前端面板(经 hub 广播)。失败不致命。"""
        if self._ws is None:
            return
        try:
            async with self._send_lock:
                await self._ws.send(json.dumps({"type": "tt.status", **payload}))
        except Exception:
            pass

    async def events(self):
        if self._ws is None:
            raise RuntimeError("observer not connected")
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "response.output.delta":
                kind = msg.get("kind")
                yield TalkerEvent(kind=kind or "unknown", text=msg.get("text", "") or "",
                                  audio_b64=msg.get("audio"), response_id=msg.get("response_id"),
                                  end_of_turn=(kind == "listen"), raw=msg)
            elif msg.get("type") == "user.audio":
                yield TalkerEvent(kind="user_audio", audio_b64=msg.get("audio"), raw=msg)
            elif msg.get("type") == "session.closed":
                yield TalkerEvent("closed", raw=msg)
                break
