"""在 worker 容器内直连 backend WS,验证 control.force_speak 能让模型真的出声。
用法(容器内): python smoke_force_speak.py
"""
import asyncio
import json
import sys

import websockets

URL = "ws://localhost:22500/backend"
TEXT = "等一下，这个方向不对，我们先把需求理清楚。"


async def main() -> int:
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.init", "payload": {"mode": "full_duplex"}}))
        created = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
        print("init ->", created.get("type"), created.get("session_id"))

        await ws.send(json.dumps({"type": "control.force_speak", "payload": {"text": TEXT}}))
        print("sent control.force_speak:", TEXT)

        got_text = got_audio = 0
        audio_bytes = 0
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                if msg.get("type") != "response.output.delta":
                    print("msg:", msg.get("type"), msg.get("reason") or msg.get("diagnostic") or "")
                    if msg.get("type") in ("session.closed",):
                        break
                    continue
                kind = msg.get("kind")
                if kind == "text":
                    got_text += 1
                    print("  text delta:", msg.get("text"))
                elif kind == "audio":
                    got_audio += 1
                    a = msg.get("audio") or ""
                    audio_bytes += len(a)
                    print(f"  audio delta #{got_audio}: b64 len={len(a)}")
                elif kind == "listen":
                    print("  listen delta (turn end)")
                    break
        except asyncio.TimeoutError:
            print("(timeout waiting for deltas)")

        # 24kHz float32: bytes ≈ b64len*3/4 ; samples = bytes/4 ; secs = samples/24000
        approx_secs = (audio_bytes * 0.75 / 4) / 24000 if audio_bytes else 0
        print(f"\nRESULT: text_deltas={got_text} audio_deltas={got_audio} ~audio_secs={approx_secs:.2f}")
        ok = got_audio > 0 and approx_secs > 0.3
        print("FORCE_SPEAK:", "OK" if ok else "NO AUDIO / TOO SHORT")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
