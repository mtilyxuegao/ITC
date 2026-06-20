"""Live demo runner(自动重连版):持续盯着 gateway,谁的会话活着就附上去。
会话断了自动等下一个,你随便刷新都不用重启。
由 35B Thinker 自动判断是否打断;对话记进 conversation.log。
运行: setsid .venv/bin/python tests/live_demo.py &
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinker_talker.config import Config
from thinker_talker.factory import make_thinker
from thinker_talker.orchestrator import Orchestrator
from thinker_talker.talker import GatewayObserver

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("live")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main():
    cfg = Config(
        thinker_provider=os.environ.get("THINKER_PROVIDER", "openai"),  # GPT 5.4 + web_search
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-5.4"),
        openai_api_key=os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY") or "",
        thinker_tick_seconds=1.5,
        cut_min_confidence=0.7,
        transcript_log_path=os.path.join(REPO, "conversation.log"),
    )
    base = "https://localhost:8006"
    log.info("live runner up (Thinker=%s/%s) — 等待浏览器会话(自动附着/重连)…",
             cfg.thinker_provider, cfg.openai_model)
    async with make_thinker(cfg) as thinker:
        while True:
            observer = GatewayObserver(base)
            sid = await observer.discover_session(timeout_s=6.0)
            if not sid:
                await asyncio.sleep(1.0)
                continue
            observer.session_id = sid
            try:
                await observer.connect()
            except Exception as e:  # noqa: BLE001
                log.warning("connect failed: %s", e)
                continue
            log.info("✅ 已附着会话 %s — 开始监听(聊跑偏话题会被打断)", sid)
            orch = Orchestrator(cfg, observer, thinker)

            async def _watch_newer():
                # 刷新会生成更新的会话:发现后主动断开旧的,触发重新附着(治"刷新挂掉")
                while not orch._stop.is_set():
                    await asyncio.sleep(2.0)
                    latest = await observer.discover_session(timeout_s=0.5)
                    if latest and latest != sid:
                        log.info("检测到更新会话 %s,切换", latest)
                        await observer.close()  # 断开当前 -> consume_talker 结束 -> 重新发现
                        return

            try:
                await asyncio.gather(orch.consume_talker(), orch.thinker_loop(),
                                     orch.asr_loop(), _watch_newer())
            except Exception as e:  # noqa: BLE001
                log.warning("session loop error: %s", e)
            finally:
                await observer.close()
            log.info("会话 %s 结束,等待下一个…", sid)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
