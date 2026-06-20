# ITC · Interaction Model 复刻项目

**简体中文** · [English](./README.md)

复刻 [Thinking Machines Lab](https://thinkingmachines.ai/) 的 **Interaction Model**（交互模型）的开源推理脚手架，同时支持 **SGLang** 与 **vLLM** 两套推理后端。

> ⚠️ **声明**：官方 `TML-Interaction-Small` 的权重与训练细节尚未公开（官方称将于 2026 年晚些时候开放有限研究预览）。本仓库是一个**复刻 / 研究用脚手架**，用于按照官方公开博客描述的架构来组织推理服务、流式会话与多模态接口。请将 `MODEL_PATH` 指向你自己训练或复刻的 checkpoint。

---

## 一、什么是 Interaction Model

传统对话式 AI 是**回合制**（turn-based）的：等用户说完一整句，模型再生成一整段回复。Interaction Model 把"实时交互"做进了模型本身，而不是在文本模型外面再套一层工程脚手架。

核心特点：

- **原生多模态**：从零开始在连续的音频、视频、文本流上联合训练，而非在文本模型之上后期拼接。
- **200ms 微回合（micro-turn）**：把时间切成 200ms 的小片，模型**一边持续接收** 200ms 的输入、**一边生成** 200ms 的输出，输入与输出作为并行的两条流交错进行，而不是严格的"你说完我再说"。
- **时间感知**：模型对"现在过了多久"有感知，能像真人对话一样把握节奏、停顿与打断。

官方首个模型 `TML-Interaction-Small`：

| 项目 | 规格 |
| --- | --- |
| 架构 | Mixture-of-Experts（MoE） |
| 总参数 | 276B |
| 激活参数 | 12B |
| 模态 | 音频 / 视频 / 文本（连续流） |

### 各模态的处理方式（据官方博客）

- **音频**：采用 `dMel` 表示，经轻量 embedding 层进入，输出端用 flow head 解码。
- **视频**：切成 `40x40` 的 patch，经 `hMLP` 编码。
- **文本**：标准 embedding / unembedding。
- 以上各模块均**与 transformer 从零联合训练**。

### 推理侧关键设计

- **流式会话（streaming session）**：客户端把每个 200ms 片段作为**独立请求**发送，推理服务端把这些片段**追加**到 GPU 显存中的一条**持久序列**上，避免反复分配显存与元数据开销。
- **batch-invariant kernel**：保持"训练器—采样器"逐比特对齐（bitwise alignment）。
- **MoE kernel**：用 `gather + gemv` 策略替代标准 grouped GEMM。

---

## 二、目录结构

```
ITC/
├── README.md          # 本文档
├── .env.example       # 环境变量示例 (复制为 .env 使用)
└── ...                # 你的模型代码 / 服务脚本
```

---

## 三、环境要求

- Python 3.10+
- CUDA 12.1+ 与匹配的 NVIDIA 驱动
- 多卡 GPU（276B MoE 建议 ≥4 卡，并按需调整张量/专家并行）
- 操作系统：Linux

安装推理后端（二选一或都装）：

```bash
# SGLang
pip install "sglang[all]"

# vLLM
pip install vllm
```

---

## 四、快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少设置 MODEL_PATH，并根据卡数调整 TP_SIZE / CUDA_VISIBLE_DEVICES
```

加载环境变量：

```bash
set -a && source .env && set +a
```

### 2. 启动推理服务

#### 方案 A：SGLang

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
  --max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE" \
  --context-length "$SGLANG_CONTEXT_LENGTH"
```

#### 方案 B：vLLM

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --enable-chunked-prefill \
  --dtype "$VLLM_DTYPE"
```

两套后端都默认暴露 **OpenAI 兼容** 的 HTTP 接口（`/v1/chat/completions` 等），监听 `http://$HOST:$PORT`。

### 3. 冒烟测试

```bash
curl http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$SERVED_MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}]}"
```

---

## 五、流式会话（micro-turn）说明

Interaction Model 的实时性来自 **200ms 微回合 + 持久序列** 的会话模型。要复刻这一行为，客户端需：

1. 把音频/视频/文本按 `CHUNK_MS`（默认 200ms）切片；
2. 为一次对话维持一个 **session id**，每个 200ms 片段作为独立请求带上同一 session id；
3. 服务端把片段持续 append 到 GPU 中的同一条序列，避免每片重新 prefill。

`.env` 中相关参数：

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `CHUNK_MS` | 每个 micro-turn 的时间片（毫秒） | 200 |
| `AUDIO_SAMPLE_RATE` | 音频采样率（Hz） | 16000 |
| `VIDEO_PATCH_SIZE` | 视频 patch 边长 | 40 |
| `SESSION_MAX_SECONDS` | 单会话最大时长（用于回收持久序列） | 600 |

> 提示：原生 SGLang / vLLM 默认是回合制的 OpenAI 接口。完整的"持久序列 + 200ms 流式"需要在后端之上实现一层会话调度（session manager），把分片请求映射到同一 KV 序列。本仓库的 `.env` 已为该层预留了配置项。

---

## 六、环境变量速查

完整列表见 `.env.example`，常用项：

| 变量 | 说明 |
| --- | --- |
| `INFER_BACKEND` | 后端选择：`sglang` 或 `vllm` |
| `MODEL_PATH` | 权重路径（本地目录或 HF 仓库名） |
| `SERVED_MODEL_NAME` | 对外暴露的模型名 |
| `CUDA_VISIBLE_DEVICES` | 可见 GPU |
| `TP_SIZE` / `VLLM_TENSOR_PARALLEL_SIZE` | 张量并行度 |
| `HOST` / `PORT` | 服务监听地址与端口 |
| `SGLANG_MEM_FRACTION_STATIC` / `VLLM_GPU_MEMORY_UTILIZATION` | 显存占用比例 |
| `CHUNK_MS` | micro-turn 时间片 |

---

## 七、参考资料

- [Interaction Models: A Scalable Approach to Human-AI Collaboration — Thinking Machines Lab（官方博客）](https://thinkingmachines.ai/blog/interaction-models/)
- [Mira Murati's Thinking Machines Lab Introduces Interaction Models — MarkTechPost](https://www.marktechpost.com/2026/05/13/mira-muratis-thinking-machines-lab-introduces-interaction-models-a-native-multimodal-architecture-for-real-time-human-ai-collaboration/)
- [Thinking Machines Lab Ships First Model With 200ms Real-Time Interaction — Unite.AI](https://www.unite.ai/thinking-machines-lab-ships-first-model-with-200ms-real-time-interaction/)
- [SGLang 文档](https://docs.sglang.ai/)
- [vLLM 文档](https://docs.vllm.ai/)
