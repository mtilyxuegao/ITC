"""完整自动闭环验证(在 host 上跑,localhost 可达 gateway:8006 与 thinker:30000)。

证明:真·Orchestrator 的 thinker_loop 自己 tick → 真·Thinker(SGLang 35B)判定 CUT →
经真·gateway 自动注入 force_speak → 浏览器端收到模型语音。**全程无人工调用 force_speak。**

唯一"喂"进去的是对话文字(种一段"跑偏方向"的上下文);决策→注入→出声整条链路都是真的、自动的。
(可靠地合成"用户语音"超出范围,故种文字;观察草稿这一步本身已在 e2e_observer.py 验证。)

运行: .venv/bin/python tests/e2e_autoloop.py
"""
import asyncio
import json
import logging
import os
import ssl
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websockets

from thinker_talker.config import Config
from thinker_talker.orchestrator import Orchestrator
from thinker_talker.talker import GatewayObserver
from thinker_talker.thinker import SGLangThinker

GW = "localhost:8006"
SSL = ssl._create_unverified_context()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("autoloop")


async def browser_sim(got_audio: asyncio.Event, stop: asyncio.Event):
    url = f"wss://{GW}/v1/realtime?mode=audio"
    async with websockets.connect(url, ssl=SSL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.init", "payload": {"mode": "full_duplex"}}))
        while not stop.is_set():
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
            except asyncio.TimeoutError:
                continue
            if msg.get("type") == "response.output.delta" and msg.get("kind") == "audio":
                log.info("[browser] received AUDIO b64len=%d", len(msg.get("audio") or ""))
                got_audio.set()
            elif msg.get("type") == "response.output.delta" and msg.get("kind") == "text":
                log.info("[browser] heard: %s", msg.get("text"))


async def main() -> int:
    got_audio, stop = asyncio.Event(), asyncio.Event()
    bt = asyncio.create_task(browser_sim(got_audio, stop))

    cfg = Config(
        thinker_base_url="http://localhost:30000/v1",
        thinker_model="Qwen/Qwen3.5-35B-A3B",
        thinker_abort_url="http://localhost:30000/abort_request",
        thinker_tick_seconds=1.0,
        cut_min_confidence=0.7,
    )
    observer = GatewayObserver(f"https://{GW}")

    async with SGLangThinker(cfg) as thinker:
        orch = Orchestrator(cfg, observer, thinker)
        # 等会话出现并连上观察
        await observer.connect()  # 内部 discover_session
        log.info("orchestrator observing session %s", observer.session_id)

        # 种一段"跑偏方向"的上下文(这是唯一人工喂入;决策与注入都是自动的)
        orch.state.add_turn("user", "我想给家里装一台永动机,这样永远不用交电费了")
        orch.state.add_turn("talker", "好主意!永动机确实能彻底解决电费问题,我这就帮你设计结构和选材。")

        # 跑真·闭环:consume_talker(观察) + thinker_loop(自动 tick→判定→注入)
        loop = asyncio.gather(orch.consume_talker(), orch.thinker_loop())
        try:
            await asyncio.wait_for(got_audio.wait(), timeout=40)
        except asyncio.TimeoutError:
            pass
        stop.set()
        loop.cancel()
        bt.cancel()
        for t in (loop, bt):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    print(f"\nRESULT: auto_loop_injected_audio={got_audio.is_set()}")
    print("AUTO-LOOP:", "OK ✅" if got_audio.is_set() else "FAIL")
    return 0 if got_audio.is_set() else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
