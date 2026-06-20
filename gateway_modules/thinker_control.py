"""Async Talker-Thinker control plane for the MiniCPM-o gateway.

The foreground MiniCPM-o worker remains the Talker. This module owns the
background Thinker lifecycle: submit, cancel on barge-in, and epoch-gated
write-back so stale results are never spoken.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol
from uuid import uuid4

import httpx


class TalkerState(str, Enum):
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class ThinkerTaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    STALE = "STALE"


@dataclass(slots=True)
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class ThinkerTask:
    query: str
    epoch: int
    task_id: str = field(default_factory=lambda: f"think_{uuid4().hex}")
    status: ThinkerTaskStatus = ThinkerTaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def mark(
        self,
        status: ThinkerTaskStatus,
        *,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.status = status
        self.updated_at = time.time()
        if result is not None:
            self.result = result
        if error is not None:
            self.error = error


@dataclass(frozen=True, slots=True)
class ThinkerResult:
    session_id: str
    task_id: str
    epoch: int
    text: str


@dataclass(slots=True)
class SessionState:
    session_id: str
    epoch: int = 0
    talker_state: TalkerState = TalkerState.LISTENING
    user_speaking: bool = False
    transcript: list[Turn] = field(default_factory=list)
    context_summary: str = ""
    thinker_task: Optional[ThinkerTask] = None

    def append_turn(self, role: str, content: str) -> Turn:
        turn = Turn(role=role, content=content)
        self.transcript.append(turn)
        return turn

    def active_task(self) -> Optional[ThinkerTask]:
        if self.thinker_task is None:
            return None
        if self.thinker_task.status in {
            ThinkerTaskStatus.PENDING,
            ThinkerTaskStatus.RUNNING,
        }:
            return self.thinker_task
        return None


HARD_QUESTION_MARKERS = (
    "analyze",
    "calculate",
    "compare",
    "design",
    "diagnose",
    "evaluate",
    "explain why",
    "figure out",
    "forecast",
    "multi-step",
    "plan",
    "prove",
    "reason",
    "research",
    "tradeoff",
    "why",
    "怎么",
    "为什么",
    "分析",
    "计算",
    "规划",
    "设计",
    "推理",
    "比较",
)


def should_escalate(text: str, *, min_chars: int = 140) -> bool:
    normalized = " ".join(text.lower().split())
    if len(normalized) >= min_chars:
        return True
    return any(marker in normalized for marker in HARD_QUESTION_MARKERS)


def build_thinker_query(
    state: SessionState,
    latest_user_text: str,
    *,
    max_turns: int = 8,
) -> str:
    recent_turns = state.transcript[-max_turns:]
    lines: list[str] = []

    if state.context_summary:
        lines.append(f"Conversation summary:\n{state.context_summary.strip()}")

    if recent_turns:
        lines.append("Recent transcript:")
        for turn in recent_turns:
            lines.append(f"- {turn.role}: {turn.content.strip()}")

    lines.append("Current user request:")
    lines.append(latest_user_text.strip())
    lines.append(
        "Return a concise conclusion for the MiniCPM-o Talker to say aloud. "
        "Do not include hidden reasoning or markdown."
    )
    return "\n".join(lines)


class ThinkerClient(Protocol):
    async def generate(
        self,
        *,
        task_id: str,
        query: str,
        transcript: list[Turn],
        cancel_event: asyncio.Event,
    ) -> str:
        ...

    async def cancel(self, task_id: str) -> None:
        ...


@dataclass(slots=True)
class EchoThinkerClient:
    delay_seconds: float = 0.25

    async def generate(
        self,
        *,
        task_id: str,
        query: str,
        transcript: list[Turn],
        cancel_event: asyncio.Event,
    ) -> str:
        deadline = asyncio.get_running_loop().time() + self.delay_seconds
        while asyncio.get_running_loop().time() < deadline:
            if cancel_event.is_set():
                raise asyncio.CancelledError
            await asyncio.sleep(0.02)
        if cancel_event.is_set():
            raise asyncio.CancelledError
        lines = [line.strip() for line in query.splitlines() if line.strip()]
        prompt_line = lines[-2] if len(lines) >= 2 else (lines[-1] if lines else query)
        return f"Thinker conclusion: {prompt_line[:260]}"

    async def cancel(self, task_id: str) -> None:
        return None


@dataclass(slots=True)
class OpenAICompatibleThinkerClient:
    base_url: str
    model: str
    api_key: Optional[str] = None
    abort_endpoint: Optional[str] = None
    timeout_seconds: float = 30.0
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "OpenAICompatibleThinkerClient":
        return cls(
            base_url=os.environ.get("THINKER_BASE_URL", "http://localhost:8000"),
            model=os.environ["THINKER_MODEL"],
            api_key=os.environ.get("THINKER_API_KEY"),
            abort_endpoint=os.environ.get("THINKER_ABORT_ENDPOINT"),
            timeout_seconds=float(os.environ.get("THINKER_TIMEOUT_SECONDS", "30")),
            temperature=float(os.environ.get("THINKER_TEMPERATURE", "0.2")),
        )

    async def generate(
        self,
        *,
        task_id: str,
        query: str,
        transcript: list[Turn],
        cancel_event: asyncio.Event,
    ) -> str:
        if cancel_event.is_set():
            raise asyncio.CancelledError

        headers = {"Content-Type": "application/json", "X-Request-Id": task_id}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the async Thinker behind a real-time MiniCPM-o voice Talker. "
                        "Return only a concise conclusion for the Talker to verbalize."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "temperature": self.temperature,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            if cancel_event.is_set():
                raise asyncio.CancelledError
            data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    async def cancel(self, task_id: str) -> None:
        if not self.abort_endpoint:
            return None

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                self.abort_endpoint,
                headers=headers,
                json={"request_id": task_id, "task_id": task_id},
            )


class ThinkerGateway:
    def __init__(
        self,
        thinker_client: ThinkerClient,
        *,
        backend_name: str,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.backend_name = backend_name
        self._client = thinker_client
        self._timeout_seconds = timeout_seconds
        self._sessions: dict[str, SessionState] = {}
        self._result_queues: dict[str, asyncio.Queue[ThinkerResult]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self.invalidated_results = 0
        self.completed_results = 0
        self.cancelled_tasks = 0

    @classmethod
    def from_env(cls) -> "ThinkerGateway":
        timeout_seconds = float(os.environ.get("THINKER_TIMEOUT_SECONDS", "8"))
        if os.environ.get("THINKER_MODEL"):
            return cls(
                OpenAICompatibleThinkerClient.from_env(),
                backend_name="openai-compatible",
                timeout_seconds=timeout_seconds,
            )
        return cls(
            EchoThinkerClient(float(os.environ.get("THINKER_ECHO_DELAY_SECONDS", "0.25"))),
            backend_name="local-echo",
            timeout_seconds=timeout_seconds,
        )

    async def get_session(self, session_id: str) -> SessionState:
        async with self._lock:
            return self._get_or_create_session(session_id)

    async def maybe_escalate(self, session_id: str, text: str) -> tuple[bool, Optional[ThinkerTask]]:
        async with self._lock:
            state = self._get_or_create_session(session_id)
            state.append_turn("user", text)
            state.user_speaking = False
            escalate = should_escalate(text)

        if not escalate:
            return False, None

        state = await self.get_session(session_id)
        query = build_thinker_query(state, text)
        task = await self.submit(session_id, query)
        return True, task

    async def submit(self, session_id: str, query: str) -> ThinkerTask:
        async with self._lock:
            state = self._get_or_create_session(session_id)
            if state.active_task() is not None:
                raise RuntimeError("session already has an active Thinker task")

            task = ThinkerTask(query=query, epoch=state.epoch)
            state.thinker_task = task
            state.talker_state = TalkerState.THINKING
            cancel_event = asyncio.Event()
            self._cancel_events[task.task_id] = cancel_event
            self._workers[task.task_id] = asyncio.create_task(
                self._run_task(session_id, task, cancel_event)
            )
            return task

    async def reescalate(self, session_id: str, query: str) -> ThinkerTask:
        await self.barge_in(session_id)
        async with self._lock:
            state = self._get_or_create_session(session_id)
            state.append_turn("user", query)
        return await self.submit(session_id, query)

    async def barge_in(self, session_id: str) -> Optional[str]:
        task_to_cancel: Optional[ThinkerTask]
        async with self._lock:
            state = self._get_or_create_session(session_id)
            state.user_speaking = True
            state.epoch += 1
            state.talker_state = TalkerState.INTERRUPTED
            self._drain_stale_results_locked(session_id, state.epoch)
            task_to_cancel = state.active_task()
            if task_to_cancel is not None:
                task_to_cancel.mark(ThinkerTaskStatus.CANCELLED)
                event = self._cancel_events.get(task_to_cancel.task_id)
                if event is not None:
                    event.set()
                self.cancelled_tasks += 1
            state.thinker_task = None
            state.talker_state = TalkerState.LISTENING

        if task_to_cancel is not None:
            await self._client.cancel(task_to_cancel.task_id)
            return task_to_cancel.task_id
        return None

    async def wait_for_result(
        self,
        session_id: str,
        *,
        timeout_seconds: float = 0.0,
        task_id: Optional[str] = None,
    ) -> Optional[ThinkerResult]:
        queue = self._queue_for(session_id)
        if task_id is not None:
            return await self._wait_for_task_result(
                queue,
                task_id,
                timeout_seconds=timeout_seconds,
            )
        try:
            if timeout_seconds <= 0:
                return queue.get_nowait()
            return await asyncio.wait_for(queue.get(), timeout=timeout_seconds)
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            return None

    async def shutdown(self) -> None:
        workers = list(self._workers.values())
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()
        self._cancel_events.clear()

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "session_count": len(self._sessions),
            "active_task_count": len(self._workers),
            "completed_results": self.completed_results,
            "cancelled_tasks": self.cancelled_tasks,
            "invalidated_results": self.invalidated_results,
        }

    def serialize_state(self, state: SessionState) -> dict[str, Any]:
        return {
            "session_id": state.session_id,
            "epoch": state.epoch,
            "talker_state": state.talker_state.value,
            "user_speaking": state.user_speaking,
            "transcript": [
                {"role": turn.role, "content": turn.content, "timestamp": turn.timestamp}
                for turn in state.transcript[-12:]
            ],
            "thinker_task": self.serialize_task(state.thinker_task),
        }

    def serialize_task(self, task: Optional[ThinkerTask]) -> Optional[dict[str, Any]]:
        if task is None:
            return None
        return {
            "task_id": task.task_id,
            "epoch": task.epoch,
            "status": task.status.value,
            "query": task.query,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
        }

    def serialize_result(self, result: Optional[ThinkerResult]) -> Optional[dict[str, Any]]:
        if result is None:
            return None
        return {
            "session_id": result.session_id,
            "task_id": result.task_id,
            "epoch": result.epoch,
            "text": result.text,
        }

    def _get_or_create_session(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
        self._queue_for(session_id)
        return state

    def _queue_for(self, session_id: str) -> asyncio.Queue[ThinkerResult]:
        queue = self._result_queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue()
            self._result_queues[session_id] = queue
        return queue

    async def _wait_for_task_result(
        self,
        queue: asyncio.Queue[ThinkerResult],
        task_id: str,
        *,
        timeout_seconds: float,
    ) -> Optional[ThinkerResult]:
        kept: list[ThinkerResult] = []
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        try:
            while True:
                try:
                    if timeout_seconds <= 0:
                        result = queue.get_nowait()
                    else:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            return None
                        result = await asyncio.wait_for(queue.get(), timeout=remaining)
                except (asyncio.QueueEmpty, asyncio.TimeoutError):
                    return None

                if result.task_id == task_id:
                    return result
                kept.append(result)
        finally:
            for result in kept:
                queue.put_nowait(result)

    def _drain_stale_results_locked(self, session_id: str, current_epoch: int) -> None:
        queue = self._queue_for(session_id)
        kept: list[ThinkerResult] = []
        while True:
            try:
                result = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if result.epoch == current_epoch:
                kept.append(result)
            else:
                self.invalidated_results += 1
        for result in kept:
            queue.put_nowait(result)

    async def _run_task(
        self,
        session_id: str,
        task: ThinkerTask,
        cancel_event: asyncio.Event,
    ) -> None:
        try:
            async with self._lock:
                state = self._sessions[session_id]
                if cancel_event.is_set() or task.epoch != state.epoch:
                    if task.status != ThinkerTaskStatus.CANCELLED:
                        task.mark(ThinkerTaskStatus.STALE)
                    self.invalidated_results += 1
                    return
                task.mark(ThinkerTaskStatus.RUNNING)
                transcript = list(state.transcript)

            text = await asyncio.wait_for(
                self._client.generate(
                    task_id=task.task_id,
                    query=task.query,
                    transcript=transcript,
                    cancel_event=cancel_event,
                ),
                timeout=self._timeout_seconds,
            )

            async with self._lock:
                state = self._sessions[session_id]
                if cancel_event.is_set() or task.epoch != state.epoch:
                    if task.status != ThinkerTaskStatus.CANCELLED:
                        task.mark(ThinkerTaskStatus.STALE)
                    self.invalidated_results += 1
                    return

                task.mark(ThinkerTaskStatus.DONE, result=text)
                state.thinker_task = task
                state.append_turn("thinker", text)
                state.talker_state = TalkerState.SPEAKING
                self.completed_results += 1
                await self._queue_for(session_id).put(
                    ThinkerResult(
                        session_id=session_id,
                        task_id=task.task_id,
                        epoch=task.epoch,
                        text=text,
                    )
                )
        except asyncio.CancelledError:
            task.mark(ThinkerTaskStatus.CANCELLED)
            return
        except asyncio.TimeoutError:
            cancel_event.set()
            await self._client.cancel(task.task_id)
            task.mark(ThinkerTaskStatus.CANCELLED, error="Thinker timed out")
        except Exception as exc:
            task.mark(ThinkerTaskStatus.FAILED, error=str(exc))
        finally:
            self._workers.pop(task.task_id, None)
            self._cancel_events.pop(task.task_id, None)
