"""In-process fakes so the whole Conductor runs end-to-end with NO GPU.

- FakeVLLM    : an OpenAI-compatible /v1/chat/completions SSE server with tool calls.
- FakeGateway : a WebSocket server emulating the MiniCPM duplex gateway WRITE seam.

Both bind ephemeral loopback ports and speak real protocols, so tests exercise the
real aiohttp / websockets code paths — not mocks of them.
"""
from .fake_vllm import FakeVLLM
from .fake_gateway import FakeGateway

__all__ = ["FakeVLLM", "FakeGateway"]
