"""duplex_force_speak —— 让 MiniCPM-o duplex 在命令下立即开口说一段指定文本。

这是 docs/architecture.html 里 CUT 的"路线 2":不靠 system prompt 的软触发,而是
在模型侧加一条确定性的"开口"路径。

实现思路(复用 streaming_generate 已验证的内部件,见
MiniCPMO45/modeling_minicpmo_unified.py 4959–5310):
  1. 若正在说话,先 feed <|turn_eos|> 收尾当前 turn(镜像 force_listen 的前处理 5005–5012)。
  2. feed <|tts_bos|> 开启一个 speak turn。
  3. 把 redirect 文本逐 token "teacher-force" 进 decoder,收集每个 token 的 hidden
     ([id, hidden, eot]),并推进 LLM KV —— 这样模型"记得"自己说过这句,后续聆听才连贯。
  4. feed <|turn_eos|> 收尾。
  5. 用现成 _convert_results_to_tts_input(hidden) → tts.generate_chunk →
     _generate_waveform_from_tokens 一次性渲染整句音频。

CUT 的"先停后说":duplex_stop(self) 停掉当前 speak turn 并 flush TTS(丢弃半句残音),
让 Talker 立刻安静;duplex_force_speak 的步骤 0 在打断时也走同一套 flush,避免新句子与
被截断那句叠音。编排层 CUT 走 force_speak(text, interrupt=True),网关补丁据此先 stop+flush
再说 redirect(见 patches/integrate_force_speak.py)。

通过 install() 以 monkeypatch 挂到(force_speak 与 stop 成对):
  - 模型类(定义了 streaming_generate 的那个)→ duplex_force_speak / duplex_stop
  - core.processors.unified.DuplexView      → force_speak / stop -> DuplexGenerateResult
  - core.processors.pytorch_backend.PyTorchBackend → duplex_force_speak / duplex_stop

⚠️ 需在真机(GPU)上做一次联调验证的几点(无法离线单测):
  (a) 整句在单次 generate_chunk(max_new_token 较大)内是否能产出完整音频;
      若 TTS 有每次调用 token 上限,改为对 hidden 分块循环 generate_chunk 即可(结构已留好)。
  (b) force_speak/stop 之后恢复 duplex 的状态连贯性(滑窗/unit 记账)。本实现手动 feed 了
      turn_eos 但未走 finalize_unit 的 </unit>/register_unit_end;若发现后续聆听异常,
      在收尾后补一次轻量 unit 收尾即可。
  (c) 已在真机源码核对(MiniCPM-o-Demo, py_backend/server.py):server 端**无上行输入队列**,
      每个 input.append 同步跑一次很短的 duplex_generate,"重复输出"由客户端持续上行驱动。
      因此 stop/flush 负责掐断"当前已经在说的那一句"(截断 turn + 重置 TTS 渲染),
      而"不再被重新触发"靠编排层 ASR 自门控 + epoch 围栏(见 orchestrator.py)。
  (d) 残余播放:已下发给浏览器、排在 StreamingPcmPlayer 队列里的音频仍会播完(约播放延迟时长)。
      真正的前端 flush 需改 vendored 的 /static/duplex/lib/realtime-session.js(minified),暂不做;
      模型侧已立即停止产生新音频,体感是"尾音很短一截后切到 redirect"。
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("minicpm_ext.force_speak")


def _tt_flush_tts(self) -> None:
    """丢弃半句 TTS 缓冲并重置 token2wav,使被打断的 turn 不再吐残音。
    只动 TTS 渲染状态,不碰 LLM KV(由调用方负责 turn_eos 收尾)。"""
    self.tts_text_start_pos = 0
    self.tts_past_key_values = None
    self.tts_current_turn_start_time = None
    self._reset_token2wav_for_new_turn()


def duplex_stop(self):
    """中断当前 speak turn 并 flush TTS,让 Talker 立刻安静(CUT 的"先停后说"之"停")。

    与 duplex_force_speak 的步骤 0 同源,但不接新的发声:
      - 若正在说话:feed <|turn_eos|> 收尾当前 turn(推进 LLM KV,保持后续聆听连贯)。
      - flush TTS:丢弃尚未渲染/未发出的半句音频。
    返回一个 listen 结果(end_of_turn),让 session 据此收尾并通知前端停播。
    """
    import torch

    start_time = time.time()
    with torch.no_grad():
        if not getattr(self, "current_turn_ended", True):
            self.total_ids.append(self.turn_eos_token_id)
            self.decoder.feed(self.decoder.embed_token(self.turn_eos_token_id))
            self.current_turn_ended = True
        _tt_flush_tts(self)
    logger.info("duplex_stop: turn ended + tts flushed")
    return self._make_generate_result(start_time, is_listen=True, end_of_turn=True)


def duplex_force_speak(self, text: str, prompt_wav_path=None, max_text_tokens: int = 256):
    """挂到模型类上。返回与 streaming_generate 相同形状的 dict(_make_generate_result)。"""
    import numpy as np  # noqa: F401  (与模型其余部分一致的局部导入风格)
    import torch

    start_time = time.time()
    text = (text or "").strip()
    if not text:
        return self._make_generate_result(start_time, is_listen=True)

    with torch.no_grad():
        # 0. 正在说话则先收尾并 flush(打断:丢弃被截断那句的 TTS 残音,避免和新句子叠音)
        if not getattr(self, "current_turn_ended", True):
            self.total_ids.append(self.turn_eos_token_id)
            self.decoder.feed(self.decoder.embed_token(self.turn_eos_token_id))
            self.current_turn_ended = True
            _tt_flush_tts(self)

        # 1. 开启 speak turn
        self.total_ids.append(self.tts_bos_token_id)
        self.decoder.feed(self.decoder.embed_token(self.tts_bos_token_id))
        self.current_turn_ended = False

        # 2. teacher-force 文本,收集 hidden
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)[:max_text_tokens]
        total_hidden_in_unit = []
        total_ids_in_unit = []
        for tid in text_ids:
            self.total_ids.append(tid)
            self.res_ids.append(tid)
            self.speak_count += 1
            _logits, hidden = self.decoder.feed(self.decoder.embed_token(tid), return_logits=True)
            total_hidden_in_unit.append([tid, hidden, False])
            total_ids_in_unit.append(tid)

        # 3. 收尾 turn
        self.total_ids.append(self.turn_eos_token_id)
        self.decoder.feed(self.decoder.embed_token(self.turn_eos_token_id))
        self.current_turn_ended = True

        generated_text = self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=True)

        if not getattr(self, "generate_audio", True):
            return self._make_generate_result(
                start_time, is_listen=False, text=generated_text,
                end_of_turn=True, n_tokens=len(total_ids_in_unit),
            )

        # 4. 整句 TTS(全新 TTS turn)
        self.tts_text_start_pos = 0
        self.tts_past_key_values = None
        self.tts_current_turn_start_time = None
        self._reset_token2wav_for_new_turn()

        tts_condition = self._convert_results_to_tts_input(total_hidden_in_unit)
        new_tokens, _ = self.model.tts.generate_chunk(
            inputs_embeds=tts_condition,
            temperature=self.tts_temperature,
            repetition_penalty=self.tts_repetition_penalty,
            eos_token=self.tts_eos_token,
            force_no_stop=False,
            max_new_token=2048,
            min_new_tokens=0,
            past_key_values=None,
            logits_processors=self.tts_logits_processors,
            text_start_pos=0,
        )
        audio_waveform = self._generate_waveform_from_tokens(
            new_tokens, prompt_wav_path, is_last_chunk=True, force_flush=True
        )

        # 收尾 TTS 状态,便于恢复 duplex
        self.tts_text_start_pos = 0
        self.tts_past_key_values = None
        self.tts_current_turn_start_time = None
        self._reset_token2wav_for_new_turn()

    logger.info("force_speak rendered %d text tokens, %d tts tokens",
                len(total_ids_in_unit), int(new_tokens.numel()))
    return self._make_generate_result(
        start_time, is_listen=False, text=generated_text,
        audio_waveform=audio_waveform, end_of_turn=True,
        n_tokens=len(total_ids_in_unit), n_tts_tokens=int(new_tokens.numel()),
    )


def duplexview_force_speak(self, text: str):
    """挂到 DuplexView。镜像 DuplexView.generate() 的 dict -> DuplexGenerateResult 转换。"""
    import base64
    import numpy as np
    import torch
    from core.schemas.duplex import DuplexGenerateResult

    result = self._model.duplex_force_speak(text)

    audio_data = None
    wav = result.get("audio_waveform") if isinstance(result, dict) else None
    if wav is not None:
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        audio_data = base64.b64encode(np.asarray(wav, dtype=np.float32).tobytes()).decode("utf-8")

    g = result.get if isinstance(result, dict) else (lambda k, d=None: d)
    return DuplexGenerateResult(
        is_listen=g("is_listen", False),
        text=g("text", ""),
        audio_data=audio_data,
        end_of_turn=g("end_of_turn", True),
        current_time=g("current_time", 0),
        n_tokens=g("n_tokens"),
        n_tts_tokens=g("n_tts_tokens"),
    )


def duplexview_stop(self):
    """挂到 DuplexView。镜像 duplexview_force_speak 的 dict -> DuplexGenerateResult 转换。
    stop 不产音频(audio_data 恒为 None),用 is_listen/end_of_turn 通知 session 收尾。"""
    from core.schemas.duplex import DuplexGenerateResult

    result = self._model.duplex_stop()
    g = result.get if isinstance(result, dict) else (lambda k, d=None: d)
    return DuplexGenerateResult(
        is_listen=g("is_listen", True),
        text=g("text", ""),
        audio_data=None,
        end_of_turn=g("end_of_turn", True),
        current_time=g("current_time", 0),
        n_tokens=g("n_tokens"),
        n_tts_tokens=g("n_tts_tokens"),
    )


def backend_force_speak(self, text: str):
    """挂到 PyTorchBackend。"""
    duplex_view = self.processor.set_duplex_mode()
    return duplex_view.force_speak(text)


def backend_stop(self):
    """挂到 PyTorchBackend。停当前 turn + flush TTS。"""
    duplex_view = self.processor.set_duplex_mode()
    return duplex_view.stop()


def _find_duplex_capability_class():
    """定义了 streaming_generate 的类(真正的实现挂这里:DuplexCapability)。"""
    from MiniCPMO45 import modeling_minicpmo_unified as M
    import inspect
    for _name, obj in inspect.getmembers(M, inspect.isclass):
        if obj.__module__ == M.__name__ and hasattr(obj, "streaming_generate"):
            return obj
    raise RuntimeError("找不到定义 streaming_generate 的类(DuplexCapability)")


def model_force_speak(self, text: str, **kw):
    """挂到 MiniCPMO(DuplexView._model 的真身)。委托给 self.duplex,
    镜像 MiniCPMO.duplex_generate → self.duplex.streaming_generate 的委托方式。"""
    if getattr(self, "duplex", None) is None:
        raise RuntimeError("duplex 尚未初始化(需先 session.init/prepare)")
    return self.duplex.duplex_force_speak(text, **kw)


def model_stop(self):
    """挂到 MiniCPMO。委托给 self.duplex.duplex_stop。"""
    if getattr(self, "duplex", None) is None:
        raise RuntimeError("duplex 尚未初始化(需先 session.init/prepare)")
    return self.duplex.duplex_stop()


def install() -> None:
    """把 force_speak 挂到 DuplexCapability(实现)+ MiniCPMO(委托)+ DuplexView + PyTorchBackend。幂等。"""
    cap_cls = _find_duplex_capability_class()
    if getattr(cap_cls, "_tt_force_speak_installed", False):
        return
    cap_cls.duplex_force_speak = duplex_force_speak           # 真正实现
    cap_cls.duplex_stop = duplex_stop                         # stop/flush 实现

    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    MiniCPMO.duplex_force_speak = model_force_speak           # 委托给 self.duplex
    MiniCPMO.duplex_stop = model_stop

    from core.processors.unified import DuplexView
    DuplexView.force_speak = duplexview_force_speak
    DuplexView.stop = duplexview_stop

    from core.processors.pytorch_backend import PyTorchBackend
    PyTorchBackend.duplex_force_speak = backend_force_speak
    PyTorchBackend.duplex_stop = backend_stop

    cap_cls._tt_force_speak_installed = True
    logger.info("force_speak+stop installed on %s(impl) / MiniCPMO(delegate) / DuplexView / PyTorchBackend",
                cap_cls.__name__)
