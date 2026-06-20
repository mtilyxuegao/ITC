"""HermesThinkingClient — drive NousResearch hermes-agent as the thinking layer.

Drop-in for ThinkingClient: same `await run(epoch, task, emit, is_current)` contract,
so the Conductor doesn't change. Internally it runs hermes' synchronous
`AIAgent.run_conversation()` on a worker thread and bridges its world to ours:

    [THINK] dispatch        -> AIAgent.run_conversation() on a thread executor
    step_callback           -> emit(MILESTONE)         (the user hears progress)
    final_response          -> emit(RESULT)
    [WAIT] / barge-in       -> agent.interrupt()       (cross-thread, by design)

hermes is OpenAI-/v1-compatible (provider="custom"), so base_url points straight at
our vLLM (e.g. http://liquid-gpu-001:8001/v1, model "qwen"). Cancellation: the
Conductor cancels the asyncio task; we catch CancelledError and call interrupt() so
the still-running worker thread stops cooperatively. The epoch guard drops anything
the winding-down thread emits late, so correctness never depends on the thread
stopping instantly.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
from typing import Awaitable, Callable, List, Optional

from .thinking import MILESTONE, RESULT

EmitFn = Callable[[str, int, str], Awaitable[None]]


def _ensure_hermes_on_path() -> None:
    if "run_agent" in sys.modules:
        return
    try:
        import run_agent  # noqa: F401
        return
    except ImportError:
        path = os.environ.get("HERMES_AGENT_PATH", os.path.expanduser("~/hermes-agent"))
        if path not in sys.path:
            sys.path.insert(0, path)


class HermesThinkingClient:
    def __init__(self, base_url: str, model: str = "qwen", api_key: str = "local",
                 enabled_toolsets: Optional[List[str]] = None, max_iterations: int = 20,
                 provider: str = "custom") -> None:
        # hermes wants the OpenAI base (…/v1); accept with or without the suffix
        self.base_url = base_url if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/") + "/v1"
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.enabled_toolsets = enabled_toolsets or []
        self.max_iterations = max_iterations
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.aborts = 0

    def _build_prompt(self, task) -> str:
        c = task.constraints
        if c.get("origin"):
            return (f"帮我查 {c.get('date','')} 从 {c['origin']} 出发最便宜的航班，"
                    f"必要时用工具检索，最后用一句话给出结论。").strip()
        if task.dialog:
            return task.dialog[-1]
        return "请简要说明你能帮我做什么。"

    def _make_agent(self, epoch: int, emit: EmitFn, is_current, loop):
        _ensure_hermes_on_path()
        from run_agent import AIAgent

        def emit_ts(kind: str, text: str) -> None:
            # called on the hermes worker thread -> hop to the Conductor's loop.
            if is_current and not is_current():
                return
            try:
                asyncio.run_coroutine_threadsafe(emit(kind, epoch, text), loop)
            except Exception:
                pass  # loop gone / shutting down — epoch guard covers correctness

        def step_cb(api_call_count: int, prev_tools) -> None:
            names = ", ".join(t.get("name", "?") for t in (prev_tools or [])) or "推理中"
            emit_ts(MILESTONE, f"第{api_call_count}步：{names}")

        return AIAgent(
            base_url=self.base_url, api_key=self.api_key, provider=self.provider,
            model=self.model, max_iterations=self.max_iterations,
            enabled_toolsets=self.enabled_toolsets,
            ephemeral_system_prompt=(
                "You are a FAST research assistant for a LIVE voice call — speed matters "
                "more than completeness. Answer in 1-2 short sentences from web search "
                "result snippets. Do at most ONE or two searches; do NOT deep-extract full "
                "pages unless absolutely required. Give the answer immediately."),
            skip_context_files=True, skip_memory=True, session_db=None,
            quiet_mode=True, save_trajectories=False,
            step_callback=step_cb,
        )

    async def run(self, epoch: int, task, emit: EmitFn, is_current=None) -> None:
        loop = asyncio.get_running_loop()
        agent = self._make_agent(epoch, emit, is_current, loop)
        prompt = self._build_prompt(task)
        await emit(MILESTONE, epoch, f"思考层（hermes）接到任务：{task.summary()}")
        fut = loop.run_in_executor(
            self._executor,
            lambda: agent.run_conversation(prompt, task_id=f"epoch{epoch}"),
        )
        try:
            result = await fut
        except asyncio.CancelledError:
            self.aborts += 1
            try:
                agent.interrupt("[WAIT] barge-in / new info")  # cross-thread stop
            except Exception:
                pass
            raise
        if is_current and not is_current():
            return
        final = ""
        if isinstance(result, dict):
            final = result.get("final_response") or ""
        await emit(RESULT, epoch, final or "（没有可用结果）")
