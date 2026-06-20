"""Gateway 旁路观察 hub —— 让编排层"旁路监听"浏览器↔worker 的实时会话。

集成进 gateway.py(见 patches/integrate_observer.py):
  - 会话建立时 register_injector(session_id, worker_ws):登记一个"把消息注入 worker"的通道;
  - worker→client 的下行里,把 text/listen 等增量 publish 给订阅者;
  - 会话结束 unregister。

对外暴露两个 FastAPI 路由(install_observer(app)):
  - GET  /observer/sessions          列出当前活跃会话 id
  - WS   /observer/{session_id}      订阅该会话的增量;并可发 control.force_speak 等消息
                                      → hub 注入到该会话的 worker(实现 AI 主动打断)

全部 fail-safe:任何异常都不影响 gateway 主转发路径。
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("minicpm_ext.observer")

_subscribers: dict[str, set] = {}   # session_id -> set[asyncio.Queue]
_injectors: dict[str, object] = {}  # session_id -> async send callable


def register_injector(session_id: str, worker_ws) -> None:
    lock = asyncio.Lock()

    async def _send(raw: str) -> None:
        async with lock:
            await worker_ws.send(raw)

    _injectors[session_id] = _send
    logger.info("observer: session %s registered", session_id)


def unregister(session_id: str) -> None:
    _injectors.pop(session_id, None)
    for q in _subscribers.get(session_id, set()):
        try:
            q.put_nowait(None)  # 结束信号
        except Exception:
            pass
    _subscribers.pop(session_id, None)


def publish(session_id: str, raw: str) -> None:
    subs = _subscribers.get(session_id)
    if not subs:
        return
    for q in list(subs):
        try:
            q.put_nowait(raw)
        except asyncio.QueueFull:
            pass  # 订阅者太慢,丢弃(观察是旁路,不阻塞主路径)


def list_sessions() -> list:
    return sorted(_injectors.keys())


def publish_user_audio(session_id: str, raw: str) -> None:
    """把客户端上行的用户音频(input.append)镜像给订阅者(编排层做 ASR)。"""
    if not _subscribers.get(session_id):
        return
    import json
    try:
        msg = json.loads(raw)
    except Exception:
        return
    if msg.get("type") != "input.append":
        return
    payload = msg.get("input") or msg.get("payload") or {}
    audio = payload.get("audio_base64") or payload.get("audio")
    if audio:
        publish(session_id, json.dumps({"type": "user.audio", "audio": audio}))


# ---- 订阅辅助(供 gateway 里直接定义的 /observer 路由调用)----
def subscribe(session_id: str):
    import asyncio
    q = asyncio.Queue(maxsize=512)
    _subscribers.setdefault(session_id, set()).add(q)
    return q


def unsubscribe(session_id: str, q) -> None:
    subs = _subscribers.get(session_id)
    if subs:
        subs.discard(q)


async def inject(session_id: str, raw: str) -> None:
    """处理观察端发来的消息:
       - tt.status(编排层的状态/决策)→ 广播给本会话其他订阅者(前端状态面板);
       - 其它(如 control.force_speak)→ 注入该会话的 worker。
    """
    import json
    try:
        mtype = json.loads(raw).get("type")
    except Exception:
        mtype = None
    if mtype == "tt.status":
        publish(session_id, raw)  # 广播给前端面板,不发给 worker
        return
    inj = _injectors.get(session_id)
    if inj is not None:
        try:
            await inj(raw)
        except Exception:
            logger.exception("observer inject failed")


def install_observer(app) -> None:
    from fastapi import WebSocket

    @app.get("/observer/sessions")
    async def _observer_sessions():  # noqa: ANN202
        return {"sessions": list_sessions()}

    @app.websocket("/observer/{session_id}")
    async def _observer_ws(ws: WebSocket, session_id: str):  # noqa: ANN202
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        _subscribers.setdefault(session_id, set()).add(q)

        async def pump_down() -> None:
            while True:
                raw = await q.get()
                if raw is None:
                    break
                await ws.send_text(raw)

        async def pump_up() -> None:
            # 观察端发来的消息(如 control.force_speak)注入到该会话的 worker
            async for raw in ws.iter_text():
                inj = _injectors.get(session_id)
                if inj is not None:
                    try:
                        await inj(raw)
                    except Exception:
                        logger.exception("observer inject failed")

        try:
            await asyncio.gather(pump_down(), pump_up())
        except Exception:
            pass
        finally:
            subs = _subscribers.get(session_id)
            if subs:
                subs.discard(q)

    logger.info("observer endpoints installed: GET /observer/sessions, WS /observer/{session_id}")
