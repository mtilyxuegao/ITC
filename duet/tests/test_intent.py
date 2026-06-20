"""Intent classification: the READ seam that decides when to [THINK] / [WAIT].

The utterance literals stay Chinese on purpose: they are the zh-STT inputs the
classifier must match.
"""
from duet.intent import IntentClassifier
from duet.state import TaskState


def test_fresh_flight_request_fires_think():
    ir = IntentClassifier().classify("订周五上海出发最便宜的航班")
    assert ir.fire_think is True
    assert ir.is_interrupt is False
    assert ir.constraints_delta.get("origin") == "PVG"
    assert ir.constraints_delta.get("date") == "Fri"


def test_interrupt_with_new_info():
    task = TaskState(constraints={"origin": "PVG", "date": "Fri"})
    ir = IntentClassifier().classify("等下，改成北京出发", task, model_speaking=True)
    assert ir.is_interrupt is True
    assert ir.is_new_info is True
    assert ir.constraints_delta.get("origin") == "PEK"
    assert ir.human.get("origin") == "北京"


def test_weather_is_knowledge_intent():
    ir = IntentClassifier().classify("今天天气怎么样")
    assert ir.fire_think is True


def test_chitchat_does_not_fire():
    ir = IntentClassifier().classify("嗯嗯好的哈哈")
    assert ir.fire_think is False
    assert bool(ir) is False
