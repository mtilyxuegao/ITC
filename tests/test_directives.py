"""parse_directive 单测:不需要任何外部依赖/网络。
运行: python -m pytest tests/test_directives.py   或   python tests/test_directives.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinker_talker.directives import Action, parse_directive  # noqa: E402


def test_plain_json_noop():
    d = parse_directive('{"action":"NOOP","reason":"good enough"}')
    assert d.action is Action.NOOP
    assert d.is_noop


def test_inject():
    d = parse_directive('{"action":"INJECT","text":"补充一点","reason":"more depth"}')
    assert d.action is Action.INJECT
    assert d.text == "补充一点"


def test_cut_with_confidence():
    d = parse_directive('{"action":"CUT","text":"等一下,方向错了","confidence":0.9}')
    assert d.action is Action.CUT
    assert d.confidence == 0.9
    assert "方向错了" in d.text


def test_fenced_json():
    d = parse_directive('```json\n{"action":"CUT","text":"stop","confidence":0.8}\n```')
    assert d.action is Action.CUT
    assert d.confidence == 0.8


def test_json_with_surrounding_text():
    d = parse_directive('Sure. {"action":"INJECT","text":"x"} done')
    assert d.action is Action.INJECT
    assert d.text == "x"


def test_confidence_clamped():
    assert parse_directive('{"action":"CUT","text":"y","confidence":5}').confidence == 1.0
    assert parse_directive('{"action":"CUT","text":"y","confidence":-3}').confidence == 0.0


def test_line_fallback():
    d = parse_directive("CUT: 别说了")
    assert d.action is Action.CUT
    assert d.text == "别说了"


def test_garbage_is_noop():
    assert parse_directive("blah blah no json").action is Action.NOOP
    assert parse_directive("").action is Action.NOOP
    assert parse_directive("   ").action is Action.NOOP


def test_unknown_action_is_noop():
    assert parse_directive('{"action":"EXPLODE","text":"x"}').action is Action.NOOP


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\n{len(fns)} tests passed")
