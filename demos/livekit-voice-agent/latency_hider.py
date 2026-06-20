"""Latency hiding + verbalization (THINKER_TALKER.md §3.4, §5, §8 item 5).

Two thin helpers, both built on AgentSession.generate_reply():

  * LatencyHider.filler()       -> immediate "let me think..." the moment we escalate.
  * Verbalizer.speak_result()   -> the SINGLE path that turns Thinker text into speech.

Why generate_reply and not say(): with openai.realtime.RealtimeModel there is no
separate TTS plugin, so session.say("literal text") produces no audio. Routing
ALL output through generate_reply keeps every utterance in the Talker's own voice
(§5 step 4: "the Thinker never speaks directly; the Talker verbalizes in one
consistent voice"). The reply stays interruptible (§5 step 5), so a barge-in over
either the filler or the conclusion still works.

Caveat (documented in the spec's open_risks): generate_reply PARAPHRASES the
conclusion rather than reading it verbatim. If exact wording is required, switch
to a half-cascade (RealtimeModel text modality + a TTS plugin so say() works).
"""
from __future__ import annotations

import logging
import random

logger = logging.getLogger("thinker-talker.speech")

_VERBALIZE_TEMPLATE = (
    "You just finished thinking. Tell the user this conclusion in your own voice, "
    "naturally and concisely, as if you worked it out yourself. Do not mention that "
    "another model produced it. Conclusion: {conclusion}"
)


class LatencyHider:
    def __init__(self, fillers: list[str]) -> None:
        self._fillers = fillers or ["Let me think about that for a second."]

    def filler(self, session):
        """Speak a short filler immediately; interruptible so the user can cut in."""
        line = random.choice(self._fillers)
        logger.info("filler: %s", line)
        return session.generate_reply(
            instructions=(
                f"Say exactly this short filler and nothing else, then stop: '{line}'. "
                "Do not try to answer the question yet."
            ),
            allow_interruptions=True,
        )


class Verbalizer:
    def speak_result(self, session, text: str, *, is_fallback: bool = False):
        """Verbalize the Thinker's conclusion (or a fallback) in the Talker's voice."""
        if is_fallback:
            logger.info("verbalizing fallback")
            return session.generate_reply(
                instructions=(
                    "Tell the user, briefly and naturally, that you're not fully sure "
                    "about that one but give your best short take."
                ),
                allow_interruptions=True,
            )
        logger.info("verbalizing conclusion (%d chars)", len(text))
        return session.generate_reply(
            instructions=_VERBALIZE_TEMPLATE.format(conclusion=text),
            allow_interruptions=True,
        )
