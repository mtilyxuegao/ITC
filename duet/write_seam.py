"""The WRITE seam — how the Conductor makes the FROZEN interaction model speak.

⚠️ THIS IS THE UNVERIFIED LINCHPIN (see the design doc's "命门" panel and
scripts/h0_spike_write_seam.py). We cannot write assistant/system text into
MiniCPM's live KV. So we INVERT it: the Conductor talks to MiniCPM exactly like a
user does — it injects a short synthetic user/system text line into the supported
input path, and MiniCPM answers it in its own voice. The model never sees a control
token; it only ever responds to (real user audio) or (synthetic user-side text).

`Speaker` is the interface the Conductor depends on. Three implementations:
  - MiniCPMWriteSeam : real WebSocket injection into the duplex gateway (hackathon).
  - RecordingSpeaker : records what was "said" — the test oracle AND the caption
                       fallback (Rung-fallback: if the WRITE seam fails, show
                       backchannel as on-screen captions instead of model speech).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

import websockets


class Speaker(Protocol):
    async def say(self, text: str, epoch: int) -> bool:
        ...


@dataclass
class RecordingSpeaker:
    """Records spoken lines. Doubles as the caption-fallback Speaker."""
    spoken: List[Tuple[int, str]] = field(default_factory=list)

    async def say(self, text: str, epoch: int) -> bool:
        self.spoken.append((epoch, text))
        return True

    def texts(self) -> List[str]:
        return [t for _, t in self.spoken]


class GatewayAdapter:
    """Builds the gateway message + recognizes the model's spoken reply.

    The DEFAULT below matches duet/fakes/fake_gateway.py. At the hackathon, read
    `py_backend/server` + the gateway WS protocol and override these two methods to
    match the REAL message that carries a user turn. That is the entire H0 spike.
    """
    def build_user_turn(self, text: str) -> str:
        return json.dumps({"type": "user_text", "text": text}, ensure_ascii=False)

    def parse(self, raw: str) -> Tuple[bool, str]:
        """Return (is_spoken_reply, spoken_text)."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return False, ""
        if msg.get("type") == "assistant_speak":
            return True, msg.get("text", "")
        return False, ""


class MiniCPMWriteSeam:
    """Inject synthetic user-side text into a live MiniCPM duplex session over WS."""

    def __init__(self, gateway_url: str, adapter: Optional[GatewayAdapter] = None,
                 ack_timeout: float = 5.0) -> None:
        self.gateway_url = gateway_url
        self.adapter = adapter or GatewayAdapter()
        self.ack_timeout = ack_timeout

    async def say(self, text: str, epoch: int) -> bool:
        """Send one synthetic user turn; return True if the model spoke back."""
        try:
            async with websockets.connect(self.gateway_url) as ws:
                await ws.send(self.adapter.build_user_turn(text))
                # await the model's spoken reply (best-effort within ack_timeout)
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=self.ack_timeout)
                        spoke, _ = self.adapter.parse(raw if isinstance(raw, str) else raw.decode())
                        if spoke:
                            return True
                except asyncio.TimeoutError:
                    return False
        except Exception:
            return False
