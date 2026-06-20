"""Unit tests for the epoch-fencing gates — runnable with NO GPU and NO session.

    python -m pytest test_thinker_talker.py -q     (or just `python test_thinker_talker.py`)

These cover the core correctness claims of the design: the epoch fence, the
write-back invalidation gate, the escalation classifier, and backchannel filtering.
"""
from __future__ import annotations

from session_state import SessionState, TalkerState, TaskStatus
from escalation import EscalationClassifier, VADClassifier
from writeback import WriteBackGate, ResultEvent


def test_epoch_fence_basic():
    s = SessionState(session_id="t")
    assert s.epoch == 0
    assert s.is_current(0)
    assert s.bump_epoch() == 1
    assert not s.is_current(0)
    assert s.is_current(1)


def test_writeback_gate_rejects_stale():
    s = SessionState(session_id="t")
    gate = WriteBackGate()
    task = s.start_task("q", s.epoch)               # epoch 0
    fresh = ResultEvent(task.task_id, 0, "answer.", is_final=True)
    assert gate.accept(fresh, s) is True

    s.bump_epoch()                                  # barge-in -> epoch 1
    stale = ResultEvent(task.task_id, 0, "answer.", is_final=True)
    assert gate.accept(stale, s) is False           # ⭐ stale result never spoken


def test_single_active_task_invariant():
    s = SessionState(session_id="t")
    s.start_task("q1", s.epoch)
    assert s.has_active_task
    # Re-escalation flow: abort old (mark cancelled) + bump + new task.
    s.thinker_task.status = TaskStatus.CANCELLED
    assert not s.has_active_task
    s.bump_epoch()
    s.clear_task()
    t2 = s.start_task("q2", s.epoch)
    assert t2.epoch == 1 and s.has_active_task


def test_escalation_classifier():
    s = SessionState(session_id="t")
    c = EscalationClassifier()
    assert c.classify("hey how are you", TalkerState.LISTENING, s).should_escalate is False
    assert c.classify("thanks!", TalkerState.LISTENING, s).should_escalate is False
    assert c.classify("why is the sky blue?", TalkerState.LISTENING, s).should_escalate is True
    assert c.classify("compare a Roth IRA and a 401k", TalkerState.LISTENING, s).should_escalate is True
    assert c.classify("what's 17% of 4230?", TalkerState.LISTENING, s).should_escalate is True
    # current-info questions escalate so the Thinker can web-search
    d = c.classify("what's the latest version of Python?", TalkerState.LISTENING, s)
    assert d.should_escalate is True and d.reason == "needs_web_search"


def test_vad_backchannel_filter():
    v = VADClassifier()
    assert v.is_substantive("yeah", is_final=True) is False
    assert v.is_substantive("uh-huh", is_final=True) is False
    assert v.is_substantive("ok", is_final=True) is False
    assert v.is_substantive("wait, I meant the other one", is_final=True) is True
    assert v.is_substantive(None, is_final=False) is True  # conservative: cancel early


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
