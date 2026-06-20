"""Epoch guard: the correctness backbone of [WAIT] -- stale-epoch output is dropped.

This is what makes interruption correct regardless of whether the vLLM abort lands
instantly.
"""
import asyncio

from duet.conductor import Conductor
from duet.events import Kind
from duet.thinking import MILESTONE, ThinkingClient
from duet.write_seam import RecordingSpeaker


def _conductor():
    speaker = RecordingSpeaker()
    # base_url is never hit; we call _on_think_emit directly.
    c = Conductor(thinking=ThinkingClient("http://127.0.0.1:1"), speaker=speaker)
    return c, speaker


def test_stale_epoch_is_dropped_current_is_spoken():
    async def body():
        c, speaker = _conductor()
        c.epoch.bump()  # current epoch -> 2
        assert c.epoch.current == 2

        # stale epoch-1 milestone: must be dropped, never spoken
        await c._on_think_emit(MILESTONE, 1, "stale-PVG")
        assert speaker.spoken == []
        dropped = c.log.of_kind(Kind.DROPPED)
        assert len(dropped) == 1 and dropped[0].epoch == 1

        # current epoch-2 milestone: spoken normally
        await c._on_think_emit(MILESTONE, 2, "fresh-PEK")
        assert speaker.spoken == [(2, "fresh-PEK")]

    asyncio.run(body())
