"""Live demo runner:旁路监听浏览器当前会话,由 35B Thinker 自动判断是否打断。
不种任何上下文 —— 上下文来自真实观察到的小模型草稿。
运行: .venv/bin/python tests/live_demo.py
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinker_talker.config import Config
from thinker_talker.orchestrator import Orchestrator
from thinker_talker.talker import GatewayObserver
from thinker_talker.thinker import SGLangThinker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("live")


async def main():
    cfg = Config(
        thinker_base_url="http://localhost:30000/v1",
        thinker_model="Qwen/Qwen3.5-35B-A3B",
        thinker_abort_url="http://localhost:30000/abort_request",
        thinker_tick_seconds=1.5,
        cut_min_confidence=0.7,
        transcript_log_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversation.log"),
    )
    observer = GatewayObserver(f"https://localhost:8006")
    async with SGLangThinker(cfg) as thinker:
        orch = Orchestrator(cfg, observer, thinker)
        await observer.connect()
        log.info("✅ observing live session: %s — 开始监听,聊跑偏话题会被打断", observer.session_id)
        await asyncio.gather(orch.consume_talker(), orch.thinker_loop())


if __name__ == "__main__":
    asyncio.run(main())
