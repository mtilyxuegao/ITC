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
    "            if msg_type == \"control.stop\":  # " + MARKER + "\n"
    "                await _tt_handle_stop(session, message)\n"
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


async def _tt_handle_stop(session, message) -> None:
    """处理 control.stop:只停不说——截断当前 speak turn + flush 模型侧 TTS,让 Talker 立刻安静。
    注:重复输出由客户端持续上行驱动(server 端无输入队列),根治靠编排层 ASR 自门控 + epoch 围栏;
    这里负责掐断"已经在说的那一句"。"""
    _tt_install_force_speak()
    async with session._op_lock:
        await session._wait_finalize()
        await asyncio.to_thread(session.backend.duplex_stop)
        # 回一个 listen,让客户端切回聆听态(flush 残余播放缓冲需改前端 vendored lib,暂不做)
        await session.send_output_delta("listen", session_id=session.session_id,
                                        response_id=session._active_response_id)
        session._active_response_id = None


async def _tt_handle_force_speak(session, message) -> None:
    """处理 control.force_speak:让 Talker 立即说出指定文本,并把音频推回前端。
    payload.interrupt=True(CUT):先 duplex_stop 截断当前(可能在重复的)那句 + flush TTS,再说 redirect;
    False(INJECT):直接接话(编排层只在 IDLE 间隙才发 INJECT,故此时通常没有在说的 turn)。"""
    _tt_install_force_speak()
    payload = _payload(message)
    text = str(payload.get("text") or "").strip()
    interrupt = bool(payload.get("interrupt"))
    if not text:
        return
    async with session._op_lock:
        await session._wait_finalize()
        if interrupt:
            # 截断 + flush:停掉当前 turn 的剩余 TTS,模型立刻不再吐残音;redirect 随后整句渲染
            await asyncio.to_thread(session.backend.duplex_stop)
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


# --- worker.py:让 :22400 转发层放行 control.force_speak,转发到 backend ---
WORKER_ANCHOR = (
    "                if msg_type == \"input.append\":\n"
    "                    await runtime.push(_input_payload(msg))\n"
    "                    continue"
)
WORKER_REP = WORKER_ANCHOR + (
    "\n\n                if msg_type in (\"control.force_speak\", \"control.stop\"):  # " + MARKER + "\n"
    "                    await runtime.backend.send_raw(msg)\n"
    "                    continue"
)

# --- runtime/backend_client.py:RemoteBackendSession 加 send_raw,把原样消息发给 backend ---
BC_ANCHOR = "    async def push(self, input_payload: Dict[str, Any]) -> None:"
BC_REP = (
    "    async def send_raw(self, message: Dict[str, Any]) -> None:  # " + MARKER + "\n"
    "        if self._ws is None or self._closed:\n"
    "            raise RuntimeError(\"backend session is not active\")\n"
    "        await self._ws.send(json.dumps(message))\n\n"
) + BC_ANCHOR


def _patch_file(path: str, anchor: str, replacement: str, marker: str = MARKER) -> str:
    """幂等地把 anchor 替换为 replacement;已含 marker 则跳过。返回状态串。"""
    if not os.path.isfile(path):
        return f"缺文件 {path}(跳过)"
    src = open(path, encoding="utf-8").read()
    if marker in src:
        return f"{os.path.basename(path)} 已含[{marker}],跳过"
    if anchor not in src:
        return f"⚠ {os.path.basename(path)} 锚点未命中(跳过)"
    if not os.path.exists(path + ".tt.bak"):
        shutil.copy(path, path + ".tt.bak")
    open(path, "w", encoding="utf-8").write(src.replace(anchor, replacement, 1))
    return f"{os.path.basename(path)} 集成完成[{marker}]"


# --- server.py:给 Talker 会话默认 prompt 前置"对不确定/时间敏感问题拖延、不编造"的规则 ---
TALKER_PROMPT_MARKER = "tt-talker-prompt"
TALKER_ANCHOR = (
    "                system_prompt_text=_coalesce(\n"
    "                    params.get(\"system_prompt\"),\n"
    "                    params.get(\"instructions\"),\n"
    "                    default=\"You are a helpful assistant.\",\n"
    "                ),"
)
TALKER_REP = (
    "                system_prompt_text=(  # " + TALKER_PROMPT_MARKER + "\n"
    "                    \"你是简洁的实时语音助手。严格遵守:0) 开口先用一句话复述用户的问题或需求、点出关键词(例如'你想知道英伟达股价对吧'),让对方知道你听到了。\"\n"
    "                    \"1) 不确定或不知道的,绝不编造,尤其不要给出具体数字、价格、日期或事实。\"\n"
    "                    \"2) 时间敏感信息(股价、汇率、天气、新闻、今天日期、最新数据等)你没有实时联网能力,不要直接报具体数值。\"\n"
    "                    \"3) 遇到这类问题,复述后只用一句话拖延(例如'让我查一下最新的'),把具体答案留到稍后,不要急着下结论。\"\n"
    "                    \"4) 回答简短、口语化、一两句话。\\n\\n\"\n"
    "                    + (_coalesce(params.get(\"system_prompt\"), params.get(\"instructions\"), default=\"\") or \"\")\n"
    "                ),"
)


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
    else:
        if not os.path.exists(sp + ".tt.bak"):
            shutil.copy(sp, sp + ".tt.bak")
            print(f"[backup] {sp}.tt.bak")
        src = src.replace(DISPATCH_ANCHOR, DISPATCH_REPLACEMENT, 1)
        # 必须插在 def main() 之前:server.py 以 `if __name__ == "__main__": main()` 结尾,
        # main() 启动 uvicorn 会阻塞,EOF 之后的定义永远不会在 import 时执行。
        main_anchor = "\ndef main() -> None:\n"
        if main_anchor in src:
            src = src.replace(main_anchor, "\n" + APPEND_BLOCK.strip("\n") + "\n\n" + main_anchor, 1)
        else:
            src = src.rstrip("\n") + "\n" + APPEND_BLOCK  # 回退(不应发生)
        open(sp, "w", encoding="utf-8").write(src)
        print("[server.py] 集成完成(分发分支 + 处理函数,插在 main() 之前)")

    # 3. 让 force_speak 走通公网路径:worker.py 转发层 + runtime backend 客户端
    print("[worker.py]", _patch_file(os.path.join(demo, "worker.py"), WORKER_ANCHOR, WORKER_REP))
    print("[backend_client.py]", _patch_file(os.path.join(demo, "runtime", "backend_client.py"), BC_ANCHOR, BC_REP))
    # 4. 给 Talker 会话默认 prompt 加"拖延/不编造"规则
    print("[talker-prompt]", _patch_file(sp, TALKER_ANCHOR, TALKER_REP, marker=TALKER_PROMPT_MARKER))
    print("\n✅ 完成。让改动生效:重建 worker 镜像或挂载改动文件(见 patches/README.md)。")


def revert(demo: str) -> None:
    for rel in ("py_backend/server.py", "worker.py", "runtime/backend_client.py"):
        p = os.path.join(demo, rel)
        bak = p + ".tt.bak"
        if os.path.exists(bak):
            shutil.copy(bak, p)
            os.remove(bak)
            print(f"[revert] 恢复 {rel}")
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
