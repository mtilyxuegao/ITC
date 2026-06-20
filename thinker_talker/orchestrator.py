"""编排层主循环:把 Talker(小)与 Thinker(大)粘起来。

两条并发任务:
  1. consume_talker(): 消费 Talker 输出事件,维护滚动上下文与 floor/epoch。
  2. thinker_loop(): 每隔 tick 秒把上下文丢给 Thinker,按返回的 NOOP/INJECT/CUT 落地。

打断(双向):
  - 人 → AI:媒体层检测到用户开口,调 on_user_barge() → epoch++ + abort Thinker + 释放麦。
  - AI → 人:Thinker 返回 CUT 且过门控 → force_speak([CUT]+text) 立即打断。
epoch 围栏同时守住"过期答案"和"过期 CUT"。
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time

from .asr import transcribe
from .config import Config
from .directives import Action, Directive
from .state import Floor, FloorArbiter, SessionState, TalkerState
from .talker import TalkerClient
from .thinker import SGLangThinker

logger = logging.getLogger("tt.orch")


class Orchestrator:
    def __init__(self, cfg: Config, talker: TalkerClient, thinker: SGLangThinker):
        self.cfg = cfg
        self.talker = talker
        self.thinker = thinker
        self.state = SessionState()
        self.arbiter = FloorArbiter(self.state)
        self._stop = asyncio.Event()
        self._tlog_path = cfg.transcript_log_path or ""
        self._last_thought_ctx = None  # 去重:上下文没变就不重复打扰大模型
        self._audio_buf = bytearray()  # 用户上行音频(float32 16k),供 ASR
        self._last_user_text = ""
        self._ai_audio_guard_until = 0.0  # AI 说话 + 尾窗:此刻之前丢弃上行音频(防自听)
        self._asr_enabled = cfg.asr_enabled and ((cfg.asr_provider == "local") or bool(cfg.asr_api_key))

    def _tlog(self, tag: str, text: str) -> None:
        """把一条对话/决策写进 conversation log(便于核对大模型是否真被调用)。"""
        if not text:
            return
        line = f"{time.strftime('%H:%M:%S')} [{tag}] {text}"
        logger.info("%s", line)
        if self._tlog_path:
            try:
                with open(self._tlog_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    # ---------- 自听门控(防 AI 输出被自己 ASR 回采成"用户输入") ----------
    def _mark_ai_speaking(self) -> None:
        """AI(Talker 草稿 / INJECT / CUT)正在出声:刷新尾窗并清掉已缓冲的上行,
        这样 AI 这段时间的音频既不会被累积,也不会在停说后被转写。"""
        self._ai_audio_guard_until = time.time() + self.cfg.asr_echo_guard_seconds
        self._audio_buf.clear()

    def _ai_holding(self) -> bool:
        """AI 是否正持麦或处于停说尾窗内(此时上行多半是自己的回声)。"""
        return (self.state.floor is not Floor.IDLE
                or self.state.talker_state is not TalkerState.LISTENING
                or time.time() < self._ai_audio_guard_until)

    # ---------- 打断:人 → AI ----------
    async def on_user_barge(self) -> None:
        """媒体层/observer 在检测到用户开口(force_listen)时调用。"""
        self.state.user_speaking = True
        new_epoch = self.state.bump_epoch()           # fence:作废一切旧 epoch 的指令
        await self.thinker.abort()                      # 源头取消在飞的 Thinker
        self.arbiter.release()
        self.state.talker_state = TalkerState.LISTENING
        logger.info("user barge-in -> epoch=%d, thinker aborted", new_epoch)

    # ---------- 消费 Talker 输出 ----------
    async def consume_talker(self) -> None:
        async for ev in self.talker.events():
            if ev.kind == "text" and ev.text:
                # 小模型草稿:既是给用户的即时回答,也是给 Thinker 的桥接素材
                self.state.add_turn("talker", ev.text)
                self._tlog("小模型", ev.text)
                self.state.talker_state = TalkerState.SPEAKING
                self._mark_ai_speaking()  # 关掉 ASR 上行,防自己的话被回采成"用户输入"
                if self.state.floor is Floor.IDLE:
                    self.arbiter.take(Floor.TALKER)
            elif ev.kind == "listen":
                # turn 边界 / 用户在听。小模型说完了 -> 释放麦
                self.state.user_speaking = False
                if self.state.floor is Floor.TALKER:
                    self.arbiter.release()
                self.state.talker_state = TalkerState.LISTENING
            elif ev.kind == "user_audio" and ev.audio_b64:
                if self._ai_holding():
                    continue  # AI 正在说话/尾窗内:丢弃上行,避免把自己的话喂给 ASR
                try:
                    self._audio_buf += base64.b64decode(ev.audio_b64)
                except Exception:
                    pass
                maxb = 12 * 16000 * 4  # 只留最近 ~12s
                if len(self._audio_buf) > maxb:
                    del self._audio_buf[:-maxb]
            elif ev.kind == "closed":
                break
        self._stop.set()

    def record_user_text(self, text: str) -> None:
        """可选:若有 ASR 旁路拿到用户原话,喂给上下文(会推进 epoch=新输入)。"""
        if text.strip():
            self.state.add_turn("user", text)

    # ---------- Thinker tick ----------
    async def thinker_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.cfg.thinker_tick_seconds)
            ctx = self.state.render_context()
            if not ctx.strip():
                continue
            # 去重:上下文无新增就不再调用大模型(避免空转/烧钱/刷屏)
            if ctx == self._last_thought_ctx:
                continue
            self._last_thought_ctx = ctx
            epoch_snapshot = self.state.snapshot_epoch()
            searches: list = []
            t0 = time.time()
            try:
                directive = await self.thinker.think(
                    ctx,
                    on_search=lambda q, r: (searches.append(q),
                                            self._tlog("大模型·搜索", f"{q} → {r[:80].replace(chr(10), ' ')}")),
                )
            except asyncio.CancelledError:
                logger.debug("thinker tick aborted")
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning("thinker error: %s", e)
                continue
            latency_ms = int((time.time() - t0) * 1000)
            _d = directive
            # 每 tick 记一行,证明大模型在被调用 + 输入 + 耗时
            self._tlog("大模型·输入", ctx.replace("\n", " | ")[-200:])
            self._tlog("大模型·决策", f"{_d.action.value}"
                       + (f" conf={_d.confidence:.2f}" if _d.action.value == "CUT" else "")
                       + (f' "{_d.text}"' if _d.text else "")
                       + (f" [{len(searches)}次搜索]" if searches else "")
                       + f" ({latency_ms}ms)"
                       + (f" — {_d.reason[:50]}" if _d.reason else ""))
            await self._send_status(stage="thinker", input=ctx, directive=_d,
                                    searches=searches, latency_ms=latency_ms)
            await self._apply_directive(directive, epoch_snapshot)

    async def _send_status(self, *, stage: str, directive=None, **kw) -> None:
        """把状态推给前端面板(若 talker 支持 send_status)。"""
        send = getattr(self.talker, "send_status", None)
        if send is None:
            return
        payload = {"stage": stage, "ts": time.time(), **kw}
        if directive is not None:
            payload.update(action=directive.action.value, text=directive.text,
                           reason=directive.reason, confidence=directive.confidence)
        try:
            await send(payload)
        except Exception:
            pass

    async def _apply_directive(self, d: Directive, epoch_snapshot: int) -> None:
        # 汇入作废:过期一律丢弃(守过期答案 + 过期 CUT)
        if not self.state.is_current(epoch_snapshot):
            logger.info("drop stale directive %s (epoch %d != %d)", d.action.value, epoch_snapshot, self.state.epoch)
            return
        if d.action is Action.NOOP:
            return

        if d.action is Action.INJECT:
            if not d.text:
                return
            if self.cfg.inject_wait_for_gap and not self.arbiter.can_inject():
                logger.info("inject deferred (floor=%s)", self.state.floor.value)
                return
            self.arbiter.take(Floor.THINKER)
            self._mark_ai_speaking()  # INJECT 也是 AI 出声,门控 ASR 防回采
            await self.talker.force_speak(d.text)
            self._tlog("→小模型说出(INJECT)", d.text)
            await self._send_status(stage="fired", action="INJECT", text=d.text)
            self.state.add_turn("talker", d.text)
            self.arbiter.release()
            return

        if d.action is Action.CUT:
            if not self._cut_allowed(d):
                return
            self.arbiter.take(Floor.THINKER)
            self._mark_ai_speaking()  # CUT 也是 AI 出声,门控 ASR 防回采
            # 路线2(force_speak)直接让模型说 redirect;interrupt=True 让 worker 先 stop+flush
            # 当前(可能在重复的)输出,再说 redirect。不加 [CUT] 前缀(已弃用的软触发方案)。
            await self.talker.force_speak(d.text, interrupt=True)
            self._tlog("→小模型打断说出(CUT)", d.text)
            await self._send_status(stage="fired", action="CUT", text=d.text)
            self.state.add_turn("talker", d.text)
            self.arbiter.release()

    def _cut_allowed(self, d: Directive) -> bool:
        if not d.text:
            return False
        if d.confidence < self.cfg.cut_min_confidence:
            logger.info("CUT suppressed: conf %.2f < %.2f", d.confidence, self.cfg.cut_min_confidence)
            return False
        if self.cfg.interrupt_priority == "human_wins" and self.state.user_speaking:
            logger.info("CUT suppressed: human is speaking (human_wins)")
            return False
        if not self.arbiter.can_cut():
            return False
        return True

    @staticmethod
    def _rms(pcm_bytes: bytes) -> float:
        import array
        f = array.array("f")
        f.frombytes(pcm_bytes[: (len(pcm_bytes) // 4) * 4])
        if not len(f):
            return 0.0
        step = max(1, len(f) // 4000)  # 抽样,降开销
        vals = f[::step]
        return (sum(x * x for x in vals) / len(vals)) ** 0.5

    async def asr_loop(self) -> None:
        """周期性把最近的用户上行音频转写成文字,作为'用户'轮次喂给大模型。"""
        if not self._asr_enabled:
            logger.info("ASR 关闭(无 OpenAI key)")
            return
        import aiohttp
        async with aiohttp.ClientSession() as s:
            while not self._stop.is_set():
                await asyncio.sleep(2.5)
                if self._ai_holding():  # AI 正在说话/尾窗内:此刻的缓冲多半是回声,跳过
                    continue
                buf = bytes(self._audio_buf)
                if len(buf) < 16000 * 4:  # < 1s
                    continue
                if self._rms(buf) < 0.006:  # 近静音不转写(避免 whisper 幻觉出字幕之类)
                    continue
                text = await transcribe(s, self.cfg.asr_api_key, buf,
                                        base_url=self.cfg.asr_base_url,
                                        model=self.cfg.asr_model,
                                        language=self.cfg.asr_language)
                if not (text and text != self._last_user_text and len(text) >= 2):
                    continue
                # 文本级自回声过滤:转写其实是 AI 自己刚说的话 -> 丢弃,不当用户输入
                if self.state.looks_like_self_echo(text, self.cfg.asr_echo_sim_threshold):
                    self._tlog("用户(ASR·丢弃回声)", text)
                    self._audio_buf.clear()
                    continue
                self._last_user_text = text
                self.state.add_turn("user", text)
                self._tlog("用户(ASR)", text)
                await self._send_status(stage="asr", text=text)

    async def run(self) -> None:
        await self.talker.connect()
        await asyncio.gather(self.consume_talker(), self.thinker_loop(), self.asr_loop())
