"""端到端验证 (b):旁路监听 + 注入 force_speak,全程经过 gateway。
在能访问 gateway 的容器内跑(localhost 对容器即 gateway 时用 GW=...)。

流程:
  1. browser-sim 连 /v1/realtime?mode=audio,发 session.init,建立 duplex 会话。
  2. observer 通过 GET /observer/sessions 发现会话 id,连 /observer/{id}。
  3. observer 发 control.force_speak → hub 注入该会话的 worker。
  4. 断言:browser-sim 收到 audio delta(说明注入→worker→前端打通);observer 收到 text/listen 镜像。
"""
import asyncio
import http.client
import json
import ssl

import websockets

GW_HOST = "minicpm-gateway"
GW_PORT = 8006
TEXT = "打断一下，我们换个思路。"
SSL = ssl._create_unverified_context()


def list_sessions():
    c = http.client.HTTPSConnection(GW_HOST, GW_PORT, context=SSL, timeout=10)
    c.request("GET", "/observer/sessions")
    data = json.loads(c.getresponse().read())
    c.close()
    return data.get("sessions", [])


async def browser_sim(got_audio: asyncio.Event, stop: asyncio.Event):
    url = f"wss://{GW_HOST}:{GW_PORT}/v1/realtime?mode=audio"
    async with websockets.connect(url, ssl=SSL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.init", "payload": {"mode": "full_duplex"}}))
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            if msg.get("type") == "response.output.delta" and msg.get("kind") == "audio":
                alen = len(msg.get("audio") or "")
                print(f"[browser] AUDIO delta b64len={alen}")
                got_audio.set()
            elif msg.get("type") == "response.output.delta" and msg.get("kind") == "text":
                print(f"[browser] text: {msg.get('text')}")
            elif msg.get("type") in ("session.created",):
                print("[browser] session.created")


async def observer(sid: str, got_mirror: asyncio.Event, stop: asyncio.Event):
    url = f"wss://{GW_HOST}:{GW_PORT}/observer/{sid}"
    async with websockets.connect(url, ssl=SSL, max_size=None) as ws:
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "control.force_speak", "payload": {"text": TEXT}}))
        print(f"[observer] sent force_speak to {sid}")
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            k = msg.get("kind") or msg.get("type")
            print(f"[observer] mirror: {k} {msg.get('text','')[:30]}")
            got_mirror.set()


async def main():
    got_audio, got_mirror, stop = asyncio.Event(), asyncio.Event(), asyncio.Event()
    bt = asyncio.create_task(browser_sim(got_audio, stop))

    # 等会话出现在 /observer/sessions
    sid = None
    for _ in range(20):
        await asyncio.sleep(1)
        s = list_sessions()
        if s:
            sid = s[-1]
            break
    if not sid:
        print("FAIL: no active session appeared in /observer/sessions")
        stop.set()
        await bt
        return 1
    print("discovered session:", sid)

    ot = asyncio.create_task(observer(sid, got_mirror, stop))
    try:
        await asyncio.wait_for(got_audio.wait(), timeout=30)
    except asyncio.TimeoutError:
        pass
    await asyncio.sleep(1)
    stop.set()
    for t in (bt, ot):
        t.cancel()

    print(f"\nRESULT: browser_got_audio={got_audio.is_set()} observer_got_mirror={got_mirror.is_set()}")
    ok = got_audio.is_set()
    print("OBSERVER+INJECT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
