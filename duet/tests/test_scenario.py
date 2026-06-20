"""End-to-end replay of the DuplexOmni figure's scenario (all over real sockets,
no GPU):

  user: "book the cheapest flight from Shanghai on Friday"
   -> model: "sure, let me check" ... (thinking layer searches PVG)
   -> user barges in: "wait, depart from Beijing instead"
   -> [CUT]+[WAIT]: cancel epoch1, epoch++, switch to PEK, re-dispatch
   -> thinking layer searches PEK -> model speaks the final result

Asserts the end-to-end correctness plus the control-signal ordering.
Chinese literals are the zh-STT inputs and the zh spoken content being matched.
"""
import asyncio

from duet.conductor import Conductor
from duet.events import Kind
from duet.fakes import FakeVLLM
from duet.state import Phase
from duet.thinking import ThinkingClient
from duet.write_seam import RecordingSpeaker


def test_flight_booking_with_barge_in_and_reset():
    async def body():
        async with FakeVLLM(chunk_delay=0.05) as fv:
            speaker = RecordingSpeaker()
            c = Conductor(thinking=ThinkingClient(fv.base_url), speaker=speaker)

            # 1) first request -> dispatch epoch 1 (search PVG)
            await c.on_user_utterance("订周五上海出发最便宜的航班")
            assert c.phase == Phase.THINKING
            assert c.epoch.current == 1
            assert c.task.constraints.get("origin") == "PVG"

            # 2) let epoch 1 stream for a bit (no result yet)
            await asyncio.sleep(0.12)

            # 3) user interrupts with new info -> [WAIT] reset -> epoch 2 (search PEK)
            await c.on_user_utterance("等下，改成北京出发")
            assert c.epoch.current == 2
            assert c.task.constraints.get("origin") == "PEK"

            # 4) drain epoch 2 to completion
            await c.drain(timeout=5.0)
            assert c.phase == Phase.LISTENING

            spoken = speaker.texts()
            joined = " | ".join(spoken)

            # final result comes from the corrected PEK task
            results = [t for _, t in speaker.spoken if "最便宜" in t]
            assert results, f"expected a final result, got: {joined}"
            assert any("CA1234" in t and "588" in t for t in results), joined

            # PVG's cheapest (MU208) must never be spoken as a result (stale epoch)
            assert "MU208" not in joined, f"stale PVG result leaked: {joined}"

            # control-signal order: THINK(1) -> CUT -> WAIT -> ... -> RESULT
            kinds = c.log.kinds()
            i_think1 = kinds.index(Kind.THINK)
            i_cut = kinds.index(Kind.CUT)
            i_wait = kinds.index(Kind.WAIT)
            i_result = len(kinds) - 1 - kinds[::-1].index(Kind.RESULT)
            assert i_think1 < i_cut < i_wait < i_result

            # two dispatches, epochs 1 then 2; final RESULT belongs to epoch 2
            think_epochs = [e.epoch for e in c.log.of_kind(Kind.THINK)]
            assert think_epochs == [1, 2], think_epochs
            assert c.log.of_kind(Kind.RESULT)[-1].epoch == 2

    asyncio.run(body())
