"""Kelly — a LiveKit voice agent (the Talker) with an async Thinker behind it.

Two-brain design per ../../docs/THINKER_TALKER.md:

  * Talker  = this AgentSession(llm=openai.realtime.RealtimeModel). Owns STT, the
              consistent voice, VAD, barge-in, and all real-time hooks. Never blocks.
  * Thinker = a large reasoning model on vLLM (separate process/GPUs), reached over
              the OpenAI-compatible API, abortable, text-in/text-out.

The wiring lives in sibling modules (config / session_state / escalation /
thinker_client / writeback / latency_hider / metrics). agent.py keeps its original
shape (AgentServer + @server.rtc_session() + RealtimeModel + cli.run_app) and only
adds the hooks that bridge the two brains.

Run (headless server, browser client): `python agent.py dev`  (needs LIVEKIT_* + OPENAI_API_KEY)
Run (local mic, no Thinker server needed for small talk): `python agent.py console`
"""
import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    StopResponse,
    cli,
)
from livekit.agents import llm
from livekit.agents.llm import function_tool
from livekit.agents import RunContext
from livekit.plugins import openai

from config import load_config
from session_state import SessionState, TalkerState
from escalation import EscalationClassifier, VADClassifier
from thinker_client import ThinkerClient
from writeback import ResultEventChannel, WriteBackGate
from latency_hider import LatencyHider, Verbalizer
from metrics import Metrics
import events

logger = logging.getLogger("thinker-talker.agent")

load_dotenv()


def _norm(state_val) -> str:
    """Normalize a LiveKit state enum/str to a lowercase string."""
    if state_val is None:
        return ""
    return (state_val.value if hasattr(state_val, "value") else str(state_val)).lower()


class MyAgent(Agent):
    def __init__(self, session_id: str = "session") -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly, a friendly voice assistant built with LiveKit. "
                "Keep your responses concise and conversational. "
                "Do not use emojis, asterisks, or markdown.\n\n"
                "You have a tool called think_deeply, backed by a powerful deep-reasoning "
                "model with live web search. Call think_deeply WHENEVER the user asks "
                "something that needs careful multi-step reasoning, math, analysis, "
                "fact-checking, or current/up-to-date information you are not certain about. "
                "When you call it, first give ONE short spoken acknowledgement (e.g. 'Let me "
                "think about that for a moment') — do NOT try to answer the hard part yourself; "
                "the detailed conclusion will be delivered and spoken automatically a moment "
                "later. For simple greetings and small talk, just answer directly without the tool."
            ),
        )
        self.cfg = load_config()
        self.state = SessionState(session_id=session_id)
        self.classifier = EscalationClassifier(min_words=self.cfg.min_words_for_length_trigger)
        self.vad = VADClassifier()
        self.thinker = ThinkerClient(self.cfg)
        self.channel = ResultEventChannel()
        self.gate = WriteBackGate()
        self.latency = LatencyHider(self.cfg.fillers)
        self.verbalizer = Verbalizer()
        self.metrics = Metrics()

        self._pump_task: asyncio.Task | None = None
        self._agent_speaking = False
        self._last_user_transcript: str | None = None
        self._buffer: list[str] = []
        self._buffer_epoch: int = -1

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user and introduce yourself briefly")
        self._pump_task = asyncio.create_task(self._result_pump())
        # Register real-time event handlers (sync callbacks; async work is spawned).
        self.session.on("user_state_changed", self._on_user_state_changed)
        self.session.on("agent_state_changed", self._on_agent_state_changed)
        self.session.on("user_input_transcribed", self._on_user_input_transcribed)
        logger.info("agent ready (escalation_enabled=%s, thinker=%s)",
                    self.cfg.escalation_enabled, self.cfg.base_url)

    async def on_exit(self) -> None:
        if self._pump_task:
            self._pump_task.cancel()
        await self.thinker.abort(self.state.thinker_task)
        self.metrics.log_summary(logger)

    # ── escalation: who decides to wake the Thinker (§3) ─────────────────────
    async def _start_think(self, question: str, reason: str):
        """Fire the Thinker asynchronously (never blocks the Talker). Shared by the
        Talker-driven tool and the heuristic auto mode. The conclusion comes back via
        _result_pump and is spoken in the Talker's voice."""
        # Re-escalation: cancel any in-flight task + advance the fence first (§3.5/§4.4).
        if self.state.has_active_task:
            await self.thinker.abort(self.state.thinker_task)
            self.state.bump_epoch()
            self.state.clear_task()

        epoch_snapshot = self.state.epoch
        task = self.state.start_task(self.state.build_query() or question, epoch_snapshot)
        self.state.talker_state = TalkerState.THINKING
        self.metrics.on_escalate()
        logger.info("escalating (reason=%s, epoch=%s, task=%s)",
                    reason, epoch_snapshot, task.task_id)
        events.emit(self.cfg, "escalated", session_id=self.state.session_id,
                    epoch=epoch_snapshot, task_id=task.task_id,
                    objective=question, reason=reason)
        asyncio.create_task(
            self.thinker.run(task, self.state, self.channel, self.metrics, question)
        )
        return task

    @function_tool
    async def think_deeply(self, context: RunContext, question: str) -> str:
        """Consult the deep-reasoning model (with live web search) for a hard or
        time-sensitive question. Call this when the user needs careful reasoning,
        multi-step analysis, math, fact-checking, or current/up-to-date information.

        Args:
            question: The user's question to reason about, in full.
        """
        if not self.cfg.escalation_enabled or self.cfg.escalation_mode == "off":
            return "Deep reasoning is currently disabled; answer from your own knowledge."
        await self._start_think(question, reason="talker_tool")
        # Return fast so the Talker never blocks. The conclusion will be spoken
        # automatically by _result_pump when the Thinker finishes.
        return ("Deep reasoning has started in the background and its conclusion will be "
                "spoken to the user automatically in a moment. Give ONE brief spoken "
                "acknowledgement now (e.g. 'Let me think about that') and do not attempt "
                "to answer the question yourself.")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        user_text = (new_message.text_content or "").strip()
        self.metrics.on_turn()
        self.state.append_turn("user", user_text)

        # Talker-driven (default): record the turn and let the Realtime model decide —
        # it will call think_deeply when it judges it necessary, else answer directly.
        if self.cfg.escalation_mode != "auto" or not self.cfg.escalation_enabled or not user_text:
            return

        # Auto mode: heuristic keyword classifier decides (legacy behavior).
        decision = self.classifier.classify(user_text, self.state.talker_state, self.state)
        if not decision.should_escalate:
            return  # small talk -> Talker self-answers (§3.1).
        await self._start_think(user_text, decision.reason)
        self.latency.filler(self.session)
        # Suppress the Realtime model's own immediate answer in auto mode.
        raise StopResponse()

    # ── barge-in: stop everything and invalidate (§4) ────────────────────────
    def _on_user_state_changed(self, ev) -> None:
        new_state = _norm(getattr(ev, "new_state", None))
        if new_state == "speaking":
            self.state.user_speaking = True
            interruptible = (
                self._agent_speaking
                or self.state.talker_state == TalkerState.THINKING
                or self.state.has_active_task
            )
            if interruptible and self.vad.is_substantive(self._last_user_transcript, False):
                asyncio.create_task(self._barge_in())
        elif new_state == "listening":
            self.state.user_speaking = False

    def _on_agent_state_changed(self, ev) -> None:
        self._agent_speaking = _norm(getattr(ev, "new_state", None)) == "speaking"

    def _on_user_input_transcribed(self, ev) -> None:
        self._last_user_transcript = getattr(ev, "transcript", None)

    async def _barge_in(self) -> None:
        """§4.1 steps ①-④: flush TTS, bump epoch, abort Thinker, snap to LISTENING."""
        self.metrics.on_interrupt()
        self.state.talker_state = TalkerState.INTERRUPTED
        try:
            self.session.interrupt(force=True)  # ① flush current speech
        except Exception as e:  # noqa: BLE001
            logger.debug("interrupt() noop: %s", e)
        self.state.bump_epoch()                  # ② advance the fence
        await self.thinker.abort(self.state.thinker_task)  # ③ cancel at source
        self.state.clear_task()
        self.state.talker_state = TalkerState.LISTENING    # ④ restart on new input
        logger.info("barge-in handled (epoch now %s)", self.state.epoch)
        events.emit(self.cfg, "interrupted", session_id=self.state.session_id,
                    epoch=self.state.epoch)

    # ── write-back: speak only epoch-valid results (§4.3, §5) ────────────────
    async def _result_pump(self) -> None:
        async for evt in self.channel.subscribe():
            # Invalidate-at-sink (gate 2): drop anything from a stale epoch.
            if not self.gate.accept(evt, self.state):
                if evt.epoch == self._buffer_epoch:
                    self._buffer.clear()
                self.metrics.on_invalidated()
                continue

            if evt.epoch != self._buffer_epoch:
                self._buffer.clear()
                self._buffer_epoch = evt.epoch

            if evt.is_fallback:
                self._buffer.clear()
                self.state.talker_state = TalkerState.SPEAKING
                self.verbalizer.speak_result(self.session, "", is_fallback=True)
                self.state.clear_task()
                self.state.talker_state = TalkerState.LISTENING
                continue

            if evt.sentence:
                self._buffer.append(evt.sentence)

            if evt.is_final:
                text = " ".join(self._buffer).strip()
                self._buffer.clear()
                # Final epoch re-check right before speaking (race-safe, §4.1-⑤).
                if text and self.state.is_current(evt.epoch):
                    self.state.talker_state = TalkerState.SPEAKING
                    self.verbalizer.speak_result(self.session, text)
                    events.emit(self.cfg, "spoken", session_id=self.state.session_id,
                                epoch=evt.epoch, task_id=evt.task_id, text=text[:300])
                    self.state.append_turn("assistant", text)
                self.state.clear_task()
                self.state.talker_state = TalkerState.LISTENING

    # ── tools (unchanged from the original demo) ─────────────────────────────
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> str:
        """Called when the user asks about the weather in a location.

        Args:
            location: The city or region the user is asking about.
        """
        logger.info(f"Looking up weather for {location}")
        return f"The weather in {location} is sunny with a temperature of 70 degrees."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # OpenAI Realtime model handles STT + LLM + TTS in one component (the Talker).
    session: AgentSession = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    await session.start(agent=MyAgent(session_id=ctx.room.name), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
