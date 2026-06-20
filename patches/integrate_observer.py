#!/usr/bin/env python3
"""把 observer hub 集成进 gateway.py(幂等),实现编排层"旁路监听"。

改 <demo>/gateway.py 四处(都 fail-safe,不影响主转发):
  1. worker_ws 建连后    -> register_injector(session_id, worker_ws)
  2. worker→client 下行  -> publish(session_id, raw)  (仅 text/listen/created/closed)
  3. finally             -> unregister(session_id)
  4. def main() 之前     -> install_observer(app)  (注册 /observer 路由)
并确保 minicpm_ext(含 observer_hub.py)已复制进 demo。

用法:
  python patches/integrate_observer.py [--demo PATH] [--check] [--revert]
应用后需重建/重启 gateway 容器才生效。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXT_SRC = os.path.join(REPO, "thinker_talker", "model_ext")
MARKER = "tt-observer"

A1 = '        worker_ws = await websockets.connect(ws_url, open_timeout=5, max_size=128 * 1024 * 1024)'
A1_REP = A1 + '''
        try:  # tt-observer
            import minicpm_ext.observer_hub as _tt_obs
            _tt_obs.register_injector(session_id, worker_ws)
        except Exception:
            pass'''

A2 = '''                    if msg.get("type") == "session.closed":
                        session_closed.set()
                        return'''
A2_REP = '''                    if msg.get("type") in ("response.output.delta", "session.created", "session.closed"):  # tt-observer
                        try:
                            import minicpm_ext.observer_hub as _tt_obs
                            if msg.get("type") != "response.output.delta" or msg.get("kind") in ("text", "listen"):
                                _tt_obs.publish(session_id, raw)
                        except Exception:
                            pass
''' + A2

A3 = '''        if worker_ws:
            try:'''
A3_REP = '''        try:  # tt-observer
            import minicpm_ext.observer_hub as _tt_obs
            _tt_obs.unregister(session_id)
        except Exception:
            pass
''' + A3

# 路由必须作为模块级 @app 装饰器定义(与 /v1/realtime 同款);
# 放在函数里 install 会导致 WS 握手被 FastAPI 拒成 403(实测)。
A4 = '@app.websocket("/v1/realtime")'
A4_REP = '''@app.get("/observer/sessions")  # tt-observer
async def _tt_observer_sessions():
    import minicpm_ext.observer_hub as _o
    return {"sessions": _o.list_sessions()}


@app.websocket("/observer/{session_id}")
async def _tt_observer_ws(ws: WebSocket, session_id: str):
    import asyncio
    import minicpm_ext.observer_hub as _o
    await ws.accept()
    q = _o.subscribe(session_id)

    async def _down():
        while True:
            raw = await q.get()
            if raw is None:
                break
            await ws.send_text(raw)

    async def _up():
        async for raw in ws.iter_text():
            await _o.inject(session_id, raw)

    try:
        await asyncio.gather(_down(), _up())
    except Exception:
        pass
    finally:
        _o.unsubscribe(session_id, q)


@app.websocket("/v1/realtime")'''


def _gw(demo: str) -> str:
    return os.path.join(demo, "gateway.py")


def check(demo: str):
    gw = _gw(demo)
    if not os.path.isfile(gw):
        return False, f"找不到 {gw}"
    src = open(gw, encoding="utf-8").read()
    if MARKER in src:
        return True, "已集成(幂等)"
    for name, a in [("A1", A1), ("A2", A2), ("A3", A3), ("A4", A4)]:
        if a not in src:
            return False, f"锚点 {name} 未命中(gateway.py 可能已变更)"
    return True, "可应用"


def apply(demo: str) -> None:
    ok, msg = check(demo)
    print("[check]", msg)
    if not ok:
        sys.exit(1)
    shutil.copytree(EXT_SRC, os.path.join(demo, "minicpm_ext"), dirs_exist_ok=True)
    print("[copy] minicpm_ext (含 observer_hub.py)")
    gw = _gw(demo)
    src = open(gw, encoding="utf-8").read()
    if MARKER in src:
        print("[gateway.py] 已集成,跳过")
        return
    if not os.path.exists(gw + ".tt.bak"):
        shutil.copy(gw, gw + ".tt.bak")
    src = src.replace(A1, A1_REP, 1)
    src = src.replace(A2, A2_REP, 1)
    src = src.replace(A3, A3_REP, 1)
    src = src.replace(A4, A4_REP, 1)
    open(gw, "w", encoding="utf-8").write(src)
    print("[gateway.py] 集成完成(4 处:register/publish/unregister/install)")
    print("\n✅ 完成。重建/重启 gateway 容器后,GET /observer/sessions、WS /observer/{id} 生效。")


def revert(demo: str) -> None:
    gw = _gw(demo)
    bak = gw + ".tt.bak"
    if os.path.exists(bak):
        shutil.copy(bak, gw)
        os.remove(bak)
        print(f"[revert] 恢复 {gw}")
    else:
        print("[revert] 无备份,跳过")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=os.path.expanduser("~/MiniCPM-o-Demo"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    if args.revert:
        revert(args.demo)
    elif args.check:
        ok, m = check(args.demo)
        print(m)
        sys.exit(0 if ok else 1)
    else:
        apply(args.demo)


if __name__ == "__main__":
    main()
