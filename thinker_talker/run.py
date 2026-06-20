"""编排层入口:python -m thinker_talker.run

读取环境变量(见 .env.thinker-talker.example),连上 Talker(gateway 或 observer)与
Thinker(GPT 或 SGLang,由 THINKER_PROVIDER 决定),启动 Orchestrator。媒体(浏览器
麦克风/摄像头)仍由原 demo 前端处理;本进程负责"草稿 → Thinker → INJECT/CUT" 控制链路。

两种 Talker 接法:
  - 设了 TALKER_OBSERVER_BASE(如 https://localhost:8006)→ GatewayObserver:旁路监听
    浏览器↔worker 的真实会话(生产用)。
  - 否则 → TalkerClient:自己当客户端驱动一个会话(本地测试用)。
"""
from __future__ import annotations

import asyncio
import logging

from .config import Config
from .factory import make_thinker
from .orchestrator import Orchestrator
from .talker import GatewayObserver, TalkerClient


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

    if cfg.talker_observer_base:
        talker = GatewayObserver(cfg.talker_observer_base)
        talker_desc = f"observer@{cfg.talker_observer_base}"
    else:
        talker = TalkerClient(cfg.talker_gateway_ws, mode=cfg.talker_session_mode)
        talker_desc = f"client@{cfg.talker_gateway_ws}"
    thinker_desc = (f"openai:{cfg.openai_model}" if cfg.thinker_provider == "openai"
                    else f"sglang:{cfg.thinker_model}")
    log.info("Talker=%s  Thinker=%s  ASR=%s", talker_desc, thinker_desc,
             "on" if cfg.asr_enabled else "off")

    async with make_thinker(cfg) as thinker:
        orch = Orchestrator(cfg, talker, thinker)
        try:
            await orch.run()
        finally:
            await talker.close()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
