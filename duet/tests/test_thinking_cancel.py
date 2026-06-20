"""The [WAIT] cancel path: cancelling the asyncio task closes the HTTP stream
(which is the abort on a real vLLM/SGLang backend).

Asserts that after cancellation no RESULT is emitted and the client recorded one
abort.
"""
import asyncio

from duet.fakes import FakeVLLM
from duet.state import TaskState
from duet.thinking import RESULT, ThinkingClient


def test_cancel_midstream_emits_no_result():
    async def body():
        async with FakeVLLM(chunk_delay=0.1) as fv:
            client = ThinkingClient(fv.base_url)
            task_state = TaskState(constraints={"origin": "PVG", "date": "Fri"})
            emits = []

            async def emit(kind, epoch, text):
                emits.append((kind, epoch, text))

            t = asyncio.create_task(client.run(1, task_state, emit, lambda: True))
            await asyncio.sleep(0.15)   # mid round-1 stream
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

            assert client.aborts == 1
            assert not any(k == RESULT for k, _, _ in emits), "no RESULT after cancel"

    asyncio.run(body())
