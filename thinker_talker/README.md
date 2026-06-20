# thinker_talker · 双脑编排层

把全双工小模型 **Talker(MiniCPM-o 4.5)** 和异步大模型 **Thinker(Qwen3.5-35B-A3B on SGLang)**
粘成一个实时系统。设计见 [`../docs/architecture.html`](../docs/architecture.html)。

## 模块

| 文件 | 作用 |
| --- | --- |
| `config.py` | 从环境变量读配置 |
| `directives.py` | Thinker 控制指令 `NOOP/INJECT/CUT` + 鲁棒解析(永不抛异常) |
| `state.py` | `SessionState`(**epoch 围栏**)+ `FloorArbiter`(谁持麦)+ 滚动上下文 |
| `prompts.py` | Thinker 系统提示(只吐一条 JSON 指令)+ Talker CUT 提示片段 |
| `thinker.py` | SGLang 客户端:`think(ctx)→Directive`,`abort()`(断连 + /abort_request) |
| `talker.py` | gateway WS 客户端:读草稿/打断,发 `control.force_speak` |
| `orchestrator.py` | 主循环:消费 Talker 事件 + Thinker tick,落地 epoch 围栏与双向打断 |
| `run.py` | 入口 `python -m thinker_talker.run` |
| `model_ext/force_speak.py` | 模型侧 `duplex_force_speak`(经 `patches/` 接入 demo) |

## 控制流(一句话)

```
Talker 草稿/打断 ──▶ Orchestrator ──ctx──▶ Thinker(SGLang)
                          │                     │
              epoch 围栏 + floor arbiter   NOOP/INJECT/CUT
                          │                     │
       force_speak / abort ◀────────────────────┘
```

- **人打断 AI**:媒体层调 `orchestrator.on_user_barge()` → `epoch++` + `thinker.abort()` + 释放麦。
- **AI 打断人**:Thinker 回 `CUT` 且过 `confidence` 门控与优先级策略 → `talker.force_speak("[CUT] …")`。
- **过期作废**:任何 directive 落地前校验 `epoch`,过期一律丢(守过期答案 + 过期 CUT)。

## 运行

```bash
pip install -r ../requirements.txt
cp ../.env.thinker-talker.example .env      # 填 SGLang 地址、模型名等
# 前置:① 小模型 demo 已跑(gateway :8006)② Qwen3.5-35B-A3B 已在 SGLang 上 serve
#       ③ 已执行 patches/integrate_force_speak.py 并让 worker 生效
python -m thinker_talker.run
```

## 测试

纯逻辑(epoch 围栏 / floor arbiter / 指令解析)无需网络/GPU:

```bash
python tests/test_directives.py
python tests/test_state.py
# 或 python -m pytest tests/
```

## 集成边界(诚实说明)

- **已验证**:编排逻辑单测、指令解析、`patches/` 能干净应用且产物可编译。
- **待真机联调**:① `force_speak` 的整句 TTS 与 duplex 状态连贯(见 `../patches/README.md`);
  ② 编排层"旁路监听"同一会话——生产里浏览器才是媒体端,需给 gateway 加一个 observer 钩子
  把草稿/打断事件转给编排层(`talker.py` 已是该 observer 的实现骨架,也可作独立测试驱动)。
- **未做**:Whisper ASR 旁路(默认用小模型草稿桥接;`orchestrator.record_user_text()` 已留接口)。
