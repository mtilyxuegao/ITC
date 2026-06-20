"""编排层入口:python -m thinker_talker.run

读取环境变量(见 .env.thinker-talker.example),连上 Talker gateway 与 SGLang Thinker,
启动 Orchestrator。媒体(浏览器麦克风/摄像头)仍由原 demo 前端处理;本进程负责
"草稿 → Thinker → INJECT/CUT" 这条控制链路。
"""
from __future__ import annotations

import asyncio
import logging

from .config import Config
from .orchestrator import Orchestrator
from .talker import TalkerClient
from .thinker import SGLangThinker


async def _main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    cfg = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("tt.run")
    log.info("Talker=%s  Thinker=%s @ %s", cfg.talker_gateway_ws, cfg.thinker_model, cfg.thinker_base_url)

    talker = TalkerClient(cfg.talker_gateway_ws, mode=cfg.talker_session_mode)
    async with SGLangThinker(cfg) as thinker:
        orch = Orchestrator(cfg, talker, thinker)
        try:
            await orch.run()
        finally:
            await talker.close()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
