"""ITC · Thinker–Talker 双脑实时交互编排层。

架构(详见 docs/architecture.html):
    Talker = MiniCPM-o 4.5 duplex(实时听/看/说,原生用户打断)
    Thinker = Qwen3.5-35B-A3B on SGLang(异步深推理,输出 NOOP/INJECT/CUT 指令)
    Orchestrator(本包)= epoch 围栏 + floor arbiter + 路由,把两者粘起来。
"""

__all__ = [
    "config",
    "directives",
    "state",
    "prompts",
    "thinker",
    "talker",
    "orchestrator",
]
