# demos —— 可跑通的参考实现

ITC 复刻的是 Thinking Machines 的 **Interaction Model**(全双工、200ms 微轮次、原生多模态流),
但官方权重尚未公开。本目录收录两个**当下就能真实跑通**的参考实现,分别对应两种交互范式,
方便对照理解 ITC 的核心概念,以及验证服务栈/推理后端选型。

| Demo | 范式 | 跑在哪 | 模型 | 一句话 |
| --- | --- | --- | --- | --- |
| [`minicpm-o-4.5-fullduplex`](./minicpm-o-4.5-fullduplex) | **全双工** | NVIDIA GPU 服务器(实测 H100) | MiniCPM-o 4.5(9B) | 公开可跑的、最接近 Interaction Model 的真·全双工多模态模型 |
| [`livekit-voice-agent`](./livekit-voice-agent) | **非双工(turn-based)** | macOS 本地 | OpenAI Realtime | 框架式语音 agent,一问一答的对照基线 |

---

## 一、双工 vs 非双工(本仓库的核心区别)

| | 非双工 / turn-based | 全双工 / full-duplex |
| --- | --- | --- |
| 类比 | 对讲机、微信语音条 | 打电话 |
| 时序 | **串行**:你说完 → 它处理 → 它回答 | **并行**:输入流与输出流互不阻塞,边听边想边说 |
| 打断 | 它说话时听不见你,得等它说完 | 你一出声即可打断;还能主动开口 |
| 实现 | LLM + 工程层拼装(STT→LLM→TTS) | 模型内生流式,滑动窗口 + 持续 KV cache |
| 典型 | 早期 Siri / 旧版语音助手 / 本仓库 LiveKit demo | TML Interaction Model / MiniCPM-o duplex 模式 |

ITC / Interaction Model 把"实时交互"做进**模型本身**(200ms 微轮次:持续吃 200ms 输入的同时生成 200ms 输出),
而不是在文本模型外面套一层工程。MiniCPM-o 4.5 的 `duplex` 模式是同一思路的公开可跑样例。

---

## 二、推理后端对 MiniCPM-o 4.5 的支持(对照 ITC 的 SGLang/vLLM 双后端)

我们查证了各框架对 MiniCPM-o 4.5 的支持现状(2026-06):

| 方案 | 支持 | 图/视频/音频**输入** | 语音(TTS)**输出** | 全双工 | 吞吐 |
| --- | --- | --- | --- | --- | --- |
| 官方 PyTorch Demo（本目录） | ✅ 官方 | ✅ | ✅ | ✅ **唯一** | 低(单会话独占卡) |
| 上游 vLLM（`vllm>=0.16`) | ✅ 仅"理解" | ✅ | ❌ | ❌ | 高 |
| SGLang | ✅ 仅"理解" | ✅ | ❌ | ❌ | 高 |
| vLLM-Omni（vllm 官方子项目 v0.22） | ✅ 全模态 | ✅ | ✅ WAV 24kHz | ❌(轮次) | 高 |

结论:
- 要**高并发多模态理解**(只要文字)→ SGLang / vLLM。
- 要**语音输出 + 高吞吐、可接受轮次**→ vLLM-Omni(`vllm serve openbmb/MiniCPM-o-4_5 --omni`)。
- 要**真·全双工实时体验**→ 只能用本目录的原生 PyTorch Demo。

**为什么全双工套不进 vLLM/SGLang**:这俩是"收齐 prompt → 批量推理"的轮次调度,
强项是纯文本 LLM 高吞吐(PagedAttention / continuous batching);
全双工需要可中断、可交替、流式喂入的自定义前向控制,套不进标准调度,只能原生 PyTorch 手写。

### 参考链接
- vLLM-Omni MiniCPM-o 示例: https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/examples/online_serving/minicpmo/
- MiniCPM-V-CookBook(vLLM 部署): https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-o4_5_vllm.md
- vllm-project/vllm-omni: https://github.com/vllm-project/vllm-omni
- SGLang 文档: https://docs.sglang.io/
- vLLM 论坛(MiniCPM-o 4.5 支持计划): https://discuss.vllm.ai/t/any-project-supported-plan-for-minicpm-o-4-5/2492

---

## 三、快速上手

- 全双工(GPU 服务器):`cd minicpm-o-4.5-fullduplex && bash deploy.sh`
- 非双工(Mac 本地):见 [`livekit-voice-agent/README.md`](./livekit-voice-agent/README.md)
