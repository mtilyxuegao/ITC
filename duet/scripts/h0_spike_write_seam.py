#!/usr/bin/env python3
"""H0 SPIKE -- the single highest-risk assumption, validated FIRST.

Question: can we inject a synthetic user/system text line into a LIVE MiniCPM duplex
session and have the model speak it back in its own voice, without resetting turn
state and without mixing audio into its mic?

Everything above Rung 1 of the design depends on this. Run it in the first hour.

  # self-test the plumbing against the in-process fake gateway:
  python -m duet.scripts.h0_spike_write_seam --demo-fake

  # the real thing at the hackathon (after wiring GatewayAdapter to the real proto):
  MINICPM_GATEWAY_URL=ws://localhost:8006/ws python -m duet.scripts.h0_spike_write_seam

Binary outcome:
  PASS -> the WRITE seam is real; proceed up the rung ladder.
  FAIL -> kill the WRITE seam now; degrade spoken backchannel to on-screen captions
          (RecordingSpeaker as a caption sink). The decoupling story still stands on
          the event log + final-result hand-back. You just saved ~15 hours.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from duet.write_seam import MiniCPMWriteSeam


async def _run(url: str, text: str, timeout: float, demo_fake: bool) -> int:
    gateway = None
    if demo_fake:
        from duet.fakes import FakeGateway
        gateway = FakeGateway()
        url = await gateway.start()
        print(f"[demo-fake] in-process gateway at {url}")

    print(f"[spike] connecting to {url}")
    print(f"[spike] injecting synthetic user turn: {text!r}")
    seam = MiniCPMWriteSeam(url, ack_timeout=timeout)
    spoke = await seam.say(text, epoch=0)

    if gateway is not None:
        print(f"[demo-fake] gateway received: {gateway.received}")
        await gateway.stop()

    print("-" * 56)
    if spoke:
        print("RESULT: PASS -- model spoke the injected line. WRITE seam is REAL.")
        print("        -> proceed up the rung ladder (Rung 1/2/3).")
        return 0
    print("RESULT: FAIL -- no spoken reply within timeout.")
    print("        -> kill the WRITE seam. Degrade backchannel to on-screen")
    print("           captions; keep the event log + final-result hand-back.")
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H0 WRITE-seam spike")
    p.add_argument("--url", default=os.environ.get("MINICPM_GATEWAY_URL", ""),
                   help="MiniCPM duplex gateway WebSocket URL")
    p.add_argument("--text", default="请用你自己的声音说：香蕉。",
                   help="synthetic user-side line to inject")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--demo-fake", action="store_true",
                   help="run against the in-process fake gateway (plumbing self-test)")
    args = p.parse_args(argv)

    if not args.url and not args.demo_fake:
        print("error: set --url or MINICPM_GATEWAY_URL, or pass --demo-fake", file=sys.stderr)
        return 2
    return asyncio.run(_run(args.url, args.text, args.timeout, args.demo_fake))


if __name__ == "__main__":
    raise SystemExit(main())
