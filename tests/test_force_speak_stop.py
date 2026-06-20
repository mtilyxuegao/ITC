"""duplex_stop / flush 的离线单测(无 GPU、无 torch):
用 FakeModel 桩验证 stop 的两件事——收尾当前 turn(feed turn_eos)+ flush TTS 状态。
运行: python tests/test_force_speak_stop.py   或   python -m pytest tests/test_force_speak_stop.py
"""
import contextlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# duplex_stop 内部 `import torch; with torch.no_grad():` —— 离线桩掉
_fake_torch = types.ModuleType("torch")
_fake_torch.no_grad = lambda: contextlib.nullcontext()
sys.modules.setdefault("torch", _fake_torch)

from thinker_talker.model_ext.force_speak import duplex_stop  # noqa: E402


class _Decoder:
    def __init__(self):
        self.fed = []

    def embed_token(self, tid):
        return ("emb", tid)

    def feed(self, emb, return_logits=False):
        self.fed.append(emb)
        return (None, ("hidden", emb)) if return_logits else ("hidden", emb)


class FakeModel:
    """只实现 duplex_stop 触达的最小接口。"""

    def __init__(self, speaking: bool):
        self.current_turn_ended = not speaking
        self.total_ids = []
        self.turn_eos_token_id = 99
        self.decoder = _Decoder()
        # 被 flush 的 TTS 状态(给非零初值,验证确实被清)
        self.tts_text_start_pos = 123
        self.tts_past_key_values = "stale-kv"
        self.tts_current_turn_start_time = 456.0
        self.reset_calls = 0

    def _reset_token2wav_for_new_turn(self):
        self.reset_calls += 1

    def _make_generate_result(self, start_time, **kw):
        return {"start_time": start_time, **kw}


def _assert_flushed(m: FakeModel):
    assert m.tts_text_start_pos == 0
    assert m.tts_past_key_values is None
    assert m.tts_current_turn_start_time is None
    assert m.reset_calls == 1, f"token2wav 应被重置一次,实际 {m.reset_calls}"


def test_stop_while_speaking_ends_turn_and_flushes():
    m = FakeModel(speaking=True)
    res = duplex_stop(m)
    # 正在说话 -> 必须 feed turn_eos 收尾
    assert m.turn_eos_token_id in m.total_ids
    assert len(m.decoder.fed) == 1
    assert m.current_turn_ended is True
    _assert_flushed(m)
    # 返回一个 listen / end_of_turn 结果,供 session 收尾
    assert res["is_listen"] is True and res["end_of_turn"] is True


def test_stop_when_idle_only_flushes():
    m = FakeModel(speaking=False)
    res = duplex_stop(m)
    # 没在说话 -> 不该再 feed turn_eos
    assert m.total_ids == []
    assert m.decoder.fed == []
    # 但 TTS 仍然 flush(掐掉可能残留的渲染状态)
    _assert_flushed(m)
    assert res["is_listen"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\n{len(fns)} tests passed")
