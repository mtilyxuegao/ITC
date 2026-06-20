"""The Conductor — pure-asyncio orchestrator that IS the control protocol.

It never touches model internals. It:
  - hears the user via its own STT transcript (`on_user_utterance`),
  - decides [THINK] / [WAIT] via the IntentClassifier,
  - dispatches NON-BLOCKING async work to the thinking layer,
  - guards every thinking-layer emission with the epoch (stale → dropped),
  - speaks back through the WRITE seam (Speaker).

State machine:
    LISTENING --intent--> DISPATCH[THINK] --> THINKING
    THINKING  --milestone--> INJECT(speak) --> THINKING
    THINKING  --result-----> INJECT(speak) --> LISTENING
    THINKING  --user new info--> [CUT]+[WAIT]: cancel+epoch++ --> DISPATCH[THINK] (new epoch)
"""
from __future__ import annotations

import asyncio
from typing import Optional

from . import thinking as T
from .events import EventLog, Kind
from .intent import IntentClassifier, IntentResult
from .state import EpochManager, Phase, TaskState
from .write_seam import Speaker


class Conductor:
    def __init__(self, thinking: T.ThinkingClient, speaker: Speaker,
                 intent: Optional[IntentClassifier] = None,
                 log: Optional[EventLog] = None, scene: str = "general",
                 immediate_ack: bool = True) -> None:
        self.thinking = thinking
        self.speaker = speaker
        self.intent = intent or IntentClassifier()
        self.log = log or EventLog()
        self.task = TaskState(scene=scene)
        self.epoch = EpochManager(1)
        self.phase = Phase.LISTENING
        self.immediate_ack = immediate_ack
        self._think_task: Optional[asyncio.Task] = None

    # ---------------------------- inputs ----------------------------
    async def on_user_utterance(self, text: str) -> IntentResult:
        """A full STT turn arrived from the Conductor's own ears."""
        self.log.emit(Kind.USER, self.epoch.current, text)
        self.task.add_user_turn(text)
        ir = self.intent.classify(text, self.task, model_speaking=(self.phase == Phase.THINKING))

        if self.phase == Phase.THINKING and ir.is_interrupt and ir.is_new_info:
            await self._wait_and_redispatch(ir)
        elif ir.fire_think:
            self.task.apply_delta(ir.constraints_delta)
            await self._dispatch(self._ack_text(ir, fresh=True))
        return ir

    def on_vad_onset(self) -> None:
        """User voice onset while the model is speaking — logged only (^)."""
        self.log.emit(Kind.ONSET, self.epoch.current)

    # --------------------------- internals ---------------------------
    def _ack_text(self, ir: IntentResult, fresh: bool) -> Optional[str]:
        if not self.immediate_ack:
            return None
        if not fresh and ir.human.get("origin"):
            return f"明白，改成从{ir.human['origin']}出发，重新查一下"
        return "好的，我查一下"

    async def _dispatch(self, ack: Optional[str]) -> None:
        ep = self.epoch.current
        self.log.emit(Kind.THINK, ep, self.task.summary())
        if ack:
            await self._speak(ack, ep, Kind.SPEAK)
        self.log.emit(Kind.DISPATCH, ep, self.task.summary())
        self.phase = Phase.THINKING
        self._think_task = asyncio.create_task(
            self.thinking.run(ep, self.task, self._on_think_emit,
                              is_current=lambda e=ep: self.epoch.is_current(e)))

    async def _on_think_emit(self, kind: str, epoch: int, text: str) -> None:
        # === EPOCH GUARD: the correctness backbone of [WAIT] ===
        if not self.epoch.is_current(epoch):
            self.log.emit(Kind.DROPPED, epoch, text, reason="stale_epoch")
            return
        if kind == T.MILESTONE:
            await self._speak(text, epoch, Kind.MILESTONE)
        elif kind == T.RESULT:
            await self._speak(text, epoch, Kind.RESULT)
            self.phase = Phase.LISTENING

    async def _wait_and_redispatch(self, ir: IntentResult) -> None:
        old = self.epoch.current
        self.log.emit(Kind.CUT, old)      # model audio stops (native duplex IRL)
        self.log.emit(Kind.WAIT, old)     # suspend/reset in-flight thinking
        await self._cancel_think()        # cancel coro + close stream
        self.epoch.bump()                 # epoch++ → any stale result is dropped
        self.task.apply_delta(ir.constraints_delta)
        await self._dispatch(self._ack_text(ir, fresh=False))

    async def _cancel_think(self) -> None:
        t = self._think_task
        if t and not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._think_task = None

    async def _speak(self, text: str, epoch: int, kind: str) -> bool:
        ok = await self.speaker.say(text, epoch)
        self.log.emit(kind, epoch, text, spoken=ok)
        return ok

    # --------------------------- test/util ---------------------------
    async def drain(self, timeout: float = 5.0) -> None:
        """Await until we return to LISTENING (or timeout). For tests/demo scripts."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while self.phase == Phase.THINKING and self._think_task is not None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(asyncio.shield(self._think_task), timeout=remaining)
            except asyncio.CancelledError:
                pass  # this episode was interrupted; loop picks up the re-dispatch
            except asyncio.TimeoutError:
                break
            except Exception:
                break
