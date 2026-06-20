"""Thinker client with mandatory abort (THINKER_TALKER.md §3, §4.2, §5, §8 item 3).

The Thinker is a large reasoning model served OpenAI-compatible by vLLM (or
SGLang). This client:

  * streams the conclusion sentence-by-sentence (§5 first-sentence optimization),
  * suppresses <think>...</think> chain-of-thought so only the conclusion is spoken,
  * enforces an ~8s timeout (§7),
  * and — critically — supports HARD ABORT (gate 1, cancel-at-source, §4.2).

### Abort mechanism (verified against current vLLM source)
vLLM V1 has **no abort HTTP endpoint**. The way to make the server stop computing
is to **cancel the asyncio task that owns the streaming request**: when the task
is cancelled, the `async with` around the stream exits, httpx closes the socket,
the server sees `http.disconnect`, and vLLM's `@with_cancellation` fires
`engine.abort(request_id)` — freeing the GPU. So `run()` records its own task and
`abort()` cancels it. (Do NOT add a Starlette BaseHTTPMiddleware to the vLLM
server — it breaks disconnect-abort: vllm-project/vllm#10087.)

SGLang exposes an explicit `POST /abort_request {"rid": ...}` which we use as a
belt-and-suspenders fallback when `backend == "sglang"`.

### Deployment caveat (verified live on this box, 2026-06)
The disconnect-abort path is a property of *recent* vLLM. On the pinned **vLLM
0.10.1.1** (forced by the server's CUDA-12.8 driver — newer vLLM ships CUDA-13
binaries), the chat-completions endpoint does NOT abort on client disconnect:
closing the response and the whole connection pool still left
`vllm:num_requests_running == 1`. So on this box `abort()` is BEST-EFFORT for the
GPU: it frees the client immediately and the epoch gate guarantees the stale result
is never spoken (safety), but the GPU keeps generating the discarded answer until
max_tokens. To get true cancel-at-source here, either upgrade the driver to CUDA 13
(unlocks latest vLLM's disconnect-abort) or run the Thinker on SGLang
(`THINKER_BACKEND=sglang`, which uses the explicit /abort_request lever above).
"""
from __future__ import annotations

import asyncio
import logging
import re

import httpx
from openai import AsyncOpenAI

from config import ThinkerConfig
from session_state import SessionState, ThinkerTask, TaskStatus
from writeback import ResultEvent, ResultEventChannel
from metrics import Metrics
import events
import tools

logger = logging.getLogger("thinker-talker.thinker")

_THINKER_SYSTEM = (
    "You are the Thinker: a deep-reasoning model standing behind a real-time "
    "voice assistant. Reason EFFICIENTLY — do not over-think simple questions. Then "
    "give a SHORT, plain-spoken conclusion (2-4 sentences) that will be read ALOUD: "
    "use plain words and spoken numbers, with NO markdown, NO bullet points, and NO "
    "LaTeX or math notation (say 'seventeen percent of 4230 is about 719', never "
    "formulas). The voice assistant says your conclusion in its own voice, so output "
    "only the spoken conclusion — not your reasoning."
)

_SENTENCE_BOUNDARY = re.compile(r"[.!?\n]")


class ThinkerClient:
    def __init__(self, cfg: ThinkerConfig) -> None:
        self.cfg = cfg
        self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY")
        self._tasks: dict[str, asyncio.Task] = {}

    async def run(
        self,
        task: ThinkerTask,
        state: SessionState,
        channel: ResultEventChannel,
        metrics: Metrics,
        question: str = "",
    ) -> None:
        """Detached entry point. Launch via asyncio.create_task(...). Optionally runs
        a web search first, then streams sentences to `channel`; enforces timeout;
        cooperatively stops on a stale epoch; aborts the vLLM request on cancellation.
        `question` is the raw user utterance, used to decide/run web search."""
        self._tasks[task.task_id] = asyncio.current_task()  # type: ignore[assignment]
        task.status = TaskStatus.RUNNING
        try:
            with metrics.time_think():
                await asyncio.wait_for(
                    self._stream(task, state, channel, question), timeout=self.cfg.timeout_s
                )
            task.status = TaskStatus.DONE
            metrics.on_thinker_done()
        except asyncio.TimeoutError:
            logger.warning("Thinker task %s timed out after %.1fs", task.task_id, self.cfg.timeout_s)
            task.status = TaskStatus.CANCELLED
            metrics.on_thinker_timeout()
            events.emit(self.cfg, "timeout", session_id=state.session_id,
                        epoch=task.epoch, task_id=task.task_id)
            # wait_for already cancelled the inner coro -> request aborted at source.
            if state.is_current(task.epoch):
                channel.publish(ResultEvent(task.task_id, task.epoch, "", is_final=True, is_fallback=True))
        except asyncio.CancelledError:
            logger.info("Thinker task %s cancelled (barge-in)", task.task_id)
            task.status = TaskStatus.CANCELLED
            await self._sglang_abort_if_needed(task)
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("Thinker task %s error: %s", task.task_id, e)
            task.status = TaskStatus.CANCELLED
            if state.is_current(task.epoch):
                channel.publish(ResultEvent(task.task_id, task.epoch, "", is_final=True, is_fallback=True))
        finally:
            self._tasks.pop(task.task_id, None)

    async def _stream(self, task: ThinkerTask, state: SessionState,
                      channel: ResultEventChannel, question: str = "") -> None:
        messages = [{"role": "system", "content": _THINKER_SYSTEM}]

        # --- Web search tool phase (THINKER_TALKER allowed_tools) ---
        if self.cfg.search_enabled and question and tools.needs_search(question):
            if not self._should_continue(task, state):
                return
            events.emit(self.cfg, "search_start", session_id=state.session_id,
                        epoch=task.epoch, task_id=task.task_id, query=question)
            results = await tools.web_search(question, self.cfg)
            events.emit(self.cfg, "search_results", session_id=state.session_id,
                        epoch=task.epoch, task_id=task.task_id, query=question, results=results)
            if results:
                messages.append({
                    "role": "system",
                    "content": "Up-to-date web search results you may use:\n"
                               + tools.format_results_for_prompt(question, results),
                })

        messages.append({"role": "user", "content": task.query})
        events.emit(self.cfg, "thinking", session_id=state.session_id,
                    epoch=task.epoch, task_id=task.task_id)

        raw = ""              # everything received
        emitted = 0           # chars of *visible* text already emitted as sentences
        last_visible = ""

        # `async with await create(...)` so __aexit__ closes the socket on cancel,
        # which triggers vLLM's disconnect-abort. NOTE: no extra_body request_id —
        # vLLM does not honor a client-supplied id in the body.
        async with await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            stream=True,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        ) as stream:
            async for chunk in stream:
                if not self._should_continue(task, state):
                    break
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if not delta:
                    continue
                raw += delta
                visible = _strip_think(raw)
                last_visible = visible

                # Emit any newly-complete sentences from the visible (post-CoT) text.
                while True:
                    m = _SENTENCE_BOUNDARY.search(visible, emitted)
                    if not m:
                        break
                    sentence = visible[emitted : m.end()].strip()
                    emitted = m.end()
                    if sentence and self._should_continue(task, state):
                        channel.publish(ResultEvent(task.task_id, task.epoch, sentence, is_final=False))

        # Flush trailing partial sentence + mark final.
        tail = last_visible[emitted:].strip()
        if tail and self._should_continue(task, state):
            channel.publish(ResultEvent(task.task_id, task.epoch, tail, is_final=False))
        if self._should_continue(task, state):
            events.emit(self.cfg, "answer", session_id=state.session_id, epoch=task.epoch,
                        task_id=task.task_id, text=last_visible.strip()[:600])
            channel.publish(ResultEvent(task.task_id, task.epoch, "", is_final=True))

    def _should_continue(self, task: ThinkerTask, state: SessionState) -> bool:
        """Cooperative abort (§4.2): stop if cancelled or the epoch moved on."""
        return task.status != TaskStatus.CANCELLED and state.is_current(task.epoch)

    async def abort(self, task: ThinkerTask | None) -> None:
        """Hard abort (gate 1): cancel the owning task so httpx disconnects and
        vLLM frees the GPU. Awaits the CancelledError so __aexit__ runs."""
        if task is None:
            return
        task.status = TaskStatus.CANCELLED
        t = self._tasks.get(task.task_id)
        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass

    async def _sglang_abort_if_needed(self, task: ThinkerTask) -> None:
        if self.cfg.backend != "sglang":
            return
        base = self.cfg.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        try:
            async with httpx.AsyncClient(timeout=2.0) as hc:
                await hc.post(f"{base}/abort_request", json={"rid": task.task_id})
        except Exception as e:  # noqa: BLE001
            logger.debug("SGLang abort_request failed (non-fatal): %s", e)


def _strip_think(raw: str) -> str:
    """Remove complete <think>...</think> blocks AND any unclosed trailing block,
    so reasoning is never emitted to speech (QwQ / R1 style CoT)."""
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<think>.*$", "", s, flags=re.DOTALL | re.IGNORECASE)
    return s
