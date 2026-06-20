"""模型侧扩展:给 MiniCPM-o duplex 加 force_speak(AI 主动打断/插话)。

这个包会被 patches/integrate_force_speak.py 复制进 demo 仓库(作为 `minicpm_ext`),
由 server.py 的 bootstrap 调用 install() 把方法挂到模型/视图/后端三个类上。
"""
from .force_speak import install  # noqa: F401
