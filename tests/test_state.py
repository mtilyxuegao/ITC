"""SessionState / epoch 围栏 / FloorArbiter 单测(无外部依赖)。
运行: python -m pytest tests/test_state.py   或   python tests/test_state.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinker_talker.state import Floor, FloorArbiter, SessionState  # noqa: E402


def test_epoch_fencing():
    s = SessionState()
    snap = s.snapshot_epoch()
    assert s.is_current(snap)
    s.bump_epoch()                    # 用户打断
    assert not s.is_current(snap)     # 旧快照作废
    assert s.is_current(s.epoch)


def test_bump_returns_new():
    s = SessionState()
    assert s.bump_epoch() == 1
    assert s.bump_epoch() == 2


def test_context_roundtrip():
    s = SessionState()
    s.add_turn("user", "今天天气")
    s.add_turn("talker", "晴天")
    s.add_turn("user", "   ")          # 空白被忽略
    ctx = s.render_context()
    assert "用户: 今天天气" in ctx
    assert "晴天" in ctx
    assert len(s.recent_context()) == 2


def test_floor_inject_only_when_idle():
    s = SessionState()
    a = FloorArbiter(s)
    assert a.can_inject()              # IDLE
    a.take(Floor.TALKER)
    assert not a.can_inject()          # 小模型在说,INJECT 要等
    a.release()
    assert a.can_inject()


def test_floor_cut_can_preempt_talker():
    s = SessionState()
    a = FloorArbiter(s)
    a.take(Floor.TALKER)
    assert a.can_cut()                 # CUT 能抢 Talker 的麦
    a.take(Floor.THINKER)
    assert not a.can_cut()             # 已在播 Thinker,不重复抢


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\n{len(fns)} tests passed")
