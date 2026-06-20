#!/usr/bin/env python3
"""把 force_speak 集成进 MiniCPM-o demo(幂等)。

做两件事:
  1. 复制 thinker_talker/model_ext/ → <demo>/minicpm_ext/(进 build context,容器内可 import)。
  2. 改 <demo>/py_backend/server.py:
     - WS 分发循环里加一个 control.force_speak 分支;
     - 文件末尾追加处理函数 + 懒加载 install()。
两处都用唯一锚点字符串定位,并带 marker 防重复。

用法:
  python patches/integrate_force_speak.py            # 应用(默认 demo=~/MiniCPM-o-Demo)
  python patches/integrate_force_speak.py --demo /path/to/MiniCPM-o-Demo
  python patches/integrate_force_speak.py --check    # 只检查能否应用,不写
  python patches/integrate_force_speak.py --revert   # 撤销
应用后需让改动生效:重建 worker 镜像,或把改动的文件挂载进容器(见 patches/README.md)。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXT_SRC = os.path.join(REPO, "thinker_talker", "model_ext")

MARKER = "thinker-talker force_speak integration"

DISPATCH_ANCHOR = (
    "            if msg_type == \"input.append\":\n"
    "                await session.push(message)\n"
    "                continue\n"
    "            # close 只走 HTTP unary 控制通道（见协议 network §3.2），WS 上不接受 close\n"
    "            raise RuntimeError(f\"unsupported message type: {msg_type}\")"
)

DISPATCH_REPLACEMENT = (
    "            if msg_type == \"input.append\":\n"
    "                await session.push(message)\n"
    "                continue\n"
    "            if msg_type == \"control.force_speak\":  # " + MARKER + "\n"
    "                await _tt_handle_force_speak(session, message)\n"
    "                continue\n"
    "            # close 只走 HTTP unary 控制通道（见协议 network §3.2），WS 上不接受 close\n"
    "            raise RuntimeError(f\"unsupported message type: {msg_type}\")"
)

APPEND_BLOCK = f'''

# ===== {MARKER} (appended by patches/integrate_force_speak.py) =====
_TT_FORCE_SPEAK_INSTALLED = False


def _tt_install_force_speak() -> None:
    global _TT_FORCE_SPEAK_INSTALLED
    if _TT_FORCE_SPEAK_INSTALLED:
        return
    from minicpm_ext.force_speak import install
    install()
    _TT_FORCE_SPEAK_INSTALLED = True


async def _tt_handle_force_speak(session, message) -> None:
    """处理 control.force_speak:让 Talker 立即说出指定文本,并把音频推回前端。"""
    _tt_install_force_speak()
    payload = _payload(message)
    text = str(payload.get("text") or "").strip()
    if not text:
        return
    async with session._op_lock:
        await session._wait_finalize()
        result = await asyncio.to_thread(session.backend.duplex_force_speak, text)
        if session._active_response_id is None:
            session._active_response_id = f"resp_{{uuid.uuid4().hex[:12]}}"
        rid = session._active_response_id
        iid = payload.get("input_id")
        if getattr(result, "text", None):
            await session.send_output_delta("text", session_id=session.session_id,
                                            response_id=rid, input_id=iid, text=result.text)
        if getattr(result, "audio_data", None):
            await session.send_output_delta("audio", session_id=session.session_id,
                                            response_id=rid, input_id=iid, audio=result.audio_data)
        await session.send_output_delta("listen", session_id=session.session_id,
                                        response_id=rid, input_id=iid)
        session._active_response_id = None
'''


def _server_path(demo: str) -> str:
    return os.path.join(demo, "py_backend", "server.py")


def check(demo: str) -> tuple[bool, str]:
    sp = _server_path(demo)
    if not os.path.isfile(sp):
        return False, f"找不到 {sp}"
    if not os.path.isdir(EXT_SRC):
        return False, f"找不到扩展源 {EXT_SRC}"
    src = open(sp, encoding="utf-8").read()
    if MARKER in src:
        return True, "已集成(marker 存在),可重复运行(幂等)"
    if DISPATCH_ANCHOR not in src:
        return False, "锚点未命中:server.py 的分发循环与预期不符(可能上游已变更)"
    return True, "可应用"


def apply(demo: str) -> None:
    ok, msg = check(demo)
    print("[check]", msg)
    if not ok:
        sys.exit(1)

    # 1. 复制扩展包
    dst = os.path.join(demo, "minicpm_ext")
    shutil.copytree(EXT_SRC, dst, dirs_exist_ok=True)
    print(f"[copy] {EXT_SRC} -> {dst}")

    # 2. 改 server.py
    sp = _server_path(demo)
    src = open(sp, encoding="utf-8").read()
    if MARKER in src:
        print("[server.py] 已集成,跳过")
        return
    if not os.path.exists(sp + ".tt.bak"):
        shutil.copy(sp, sp + ".tt.bak")
        print(f"[backup] {sp}.tt.bak")
    src = src.replace(DISPATCH_ANCHOR, DISPATCH_REPLACEMENT, 1)
    src = src.rstrip("\n") + "\n" + APPEND_BLOCK
    open(sp, "w", encoding="utf-8").write(src)
    print("[server.py] 集成完成(分发分支 + 处理函数)")
    print("\n✅ 完成。让改动生效:重建 worker 镜像或挂载改动文件(见 patches/README.md)。")


def revert(demo: str) -> None:
    sp = _server_path(demo)
    bak = sp + ".tt.bak"
    if os.path.exists(bak):
        shutil.copy(bak, sp)
        os.remove(bak)
        print(f"[revert] 恢复 {sp}")
    else:
        print("[revert] 没有备份,跳过 server.py")
    dst = os.path.join(demo, "minicpm_ext")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
        print(f"[revert] 删除 {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=os.path.expanduser("~/MiniCPM-o-Demo"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        revert(args.demo)
    elif args.check:
        ok, msg = check(args.demo)
        print(msg)
        sys.exit(0 if ok else 1)
    else:
        apply(args.demo)


if __name__ == "__main__":
    main()
