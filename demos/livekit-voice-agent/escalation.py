"""Escalation classifier + backchannel discriminator (THINKER_TALKER.md §3, §8 items 2 & 6).

Heuristic-first so it is instant (<1ms) and never blocks the Talker's real-time
loop. The call sites are stable, so a Talker-LLM classifier can replace these
heuristics later without touching agent.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from session_state import SessionState, TalkerState
from tools import needs_search

# Cues that a turn needs real reasoning / fact-check / planning / tools (§3 decide).
_REASONING_PATTERNS = re.compile(
    r"\b(why|how (do|does|can|would|should)|explain|compare|difference between|"
    r"calculate|compute|how much|how many|plan|design|analyze|evaluate|pros and cons|"
    r"look up|search|find out|figure out|work out|step by step|which is (better|cheaper|faster)|"
    r"what if|estimate|trade[- ]?off)\b",
    re.IGNORECASE,
)

_EXPLICIT_THINK = re.compile(
    r"\b(think about (this|it)|reason through|deep dive|break (this|it) down)\b",
    re.IGNORECASE,
)

# Short acknowledgements that must NOT count as a substantive interrupt (§4.4).
BACKCHANNEL_TOKENS = {
    "uh", "uh-huh", "uhhuh", "mm", "mmhm", "mhm", "hmm", "yeah", "yep", "yes",
    "ok", "okay", "right", "sure", "cool", "nice", "wow", "oh", "huh", "got it",
    "i see", "exactly", "true", "alright", "gotcha",
}


@dataclass
class EscalationDecision:
    should_escalate: bool
    query: str | None = None
    reason: str = ""


class EscalationClassifier:
    def __init__(self, min_words: int = 12) -> None:
        self.min_words = min_words

    def classify(
        self, user_text: str, talker_state: TalkerState, state: SessionState
    ) -> EscalationDecision:
        text = (user_text or "").strip()
        if not text:
            return EscalationDecision(False, reason="empty")

        if _EXPLICIT_THINK.search(text):
            return EscalationDecision(True, state.build_query(), "explicit_think")

        # Current/external-info questions: wake the Thinker so it can web-search.
        if needs_search(text):
            return EscalationDecision(True, state.build_query(), "needs_web_search")

        if _REASONING_PATTERNS.search(text):
            return EscalationDecision(True, state.build_query(), "reasoning_cue")

        # Math / numeric questions almost always benefit from the Thinker.
        if "?" in text and re.search(r"\d", text):
            return EscalationDecision(True, state.build_query(), "numeric_question")

        # Long, complex multi-clause utterances.
        if len(text.split()) >= self.min_words and "?" in text:
            return EscalationDecision(True, state.build_query(), "long_question")

        return EscalationDecision(False, reason="small_talk")


class VADClassifier:
    """Decide whether detected user speech is a *substantive* barge-in (§4.4)."""

    def is_substantive(self, transcript: str | None, is_final: bool) -> bool:
        # No transcript yet (user just started): be conservative and treat as
        # substantive so we cancel the Thinker early — cheap + safe (§4.2 notes
        # cancellation saves GPU). A pure backchannel will be re-classified once
        # its transcript arrives.
        if not transcript:
            return True
        norm = transcript.strip().lower().rstrip(".!?,")
        if norm in BACKCHANNEL_TOKENS:
            return False
        # Single very short token that is an acknowledgement.
        if len(norm.split()) <= 1 and norm in BACKCHANNEL_TOKENS:
            return False
        return True
