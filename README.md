# ITC · Interaction Model Reproduction

[简体中文](./README_zh.md) · **English**

An open-source inference scaffold for reproducing [Thinking Machines Lab](https://thinkingmachines.ai/)'s **Interaction Model**, with support for both **SGLang** and **vLLM** serving backends.

> ⚠️ **Disclaimer**: The official `TML-Interaction-Small` weights and training details are not yet public (Thinking Machines Lab states a limited research preview will open later in 2026). This repository is a **reproduction / research scaffold** that organizes the serving stack, streaming sessions, and multimodal interfaces according to the architecture described in the official blog. Point `MODEL_PATH` at your own trained or reproduced checkpoint.

---

## 1. What Is an Interaction Model

Traditional conversational AI is **turn-based**: it waits for you to finish a complete utterance, then generates a complete reply. Interaction Models bake real-time interactivity into the model itself, rather than bolting an engineering layer on top of a text model.

Key characteristics:

- **Natively multimodal**: trained from scratch jointly on continuous audio, video, and text streams, rather than stitched onto a text model after the fact.
- **200ms micro-turns**: time is sliced into 200ms chunks. The model **continuously ingests** 200ms of input while **generating** 200ms of output — input and output run as two interleaved parallel streams instead of a strict "you finish, then I speak."
- **Time awareness**: the model has a sense of "how much time has passed," letting it manage pacing, pauses, and interruptions like a real human conversation.

The first official model, `TML-Interaction-Small`:

| Item | Spec |
| --- | --- |
| Architecture | Mixture-of-Experts (MoE) |
| Total parameters | 276B |
| Active parameters | 12B |
| Modalities | Audio / video / text (continuous streams) |

### How Each Modality Is Processed (per the official blog)

- **Audio**: a `dMel` representation entering through a lightweight embedding layer, decoded on the output side by a flow head.
- **Video**: split into `40x40` patches, encoded through `hMLP`.
- **Text**: standard embedding / unembedding.
- All modules are **co-trained from scratch together with the transformer**.

### Key Inference-Side Design

- **Streaming sessions**: the client sends each 200ms chunk as a **separate request**, and the inference server **appends** these chunks into a **persistent sequence** held in GPU memory, avoiding repeated allocations and metadata overhead.
- **Batch-invariant kernels**: preserve bitwise trainer–sampler alignment.
- **MoE kernels**: use a `gather + gemv` strategy instead of standard grouped GEMM.

---

## 2. Repository Layout

```
ITC/
├── README.md          # This document (English)
├── README_zh.md       # Chinese version
├── .env.example       # Environment variable template (copy to .env)
└── ...                # Your model code / serving scripts
```

---

## 3. Requirements

- Python 3.10+
- CUDA 12.1+ with a matching NVIDIA driver
- Multi-GPU (a 276B MoE suggests ≥4 GPUs; tune tensor/expert parallelism accordingly)
- OS: Linux

Install a serving backend (either or both):

```bash
# SGLang
pip install "sglang[all]"

# vLLM
pip install vllm
```

---

## 4. Quick Start

### 1. Configure environment variables

```bash
cp .env.example .env
# Edit .env: at minimum set MODEL_PATH, and adjust TP_SIZE / CUDA_VISIBLE_DEVICES for your GPU count
```

Load the variables:

```bash
set -a && source .env && set +a
```

### 2. Launch the serving stack

#### Option A: SGLang

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

#### Option B: vLLM

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

Both backends expose an **OpenAI-compatible** HTTP API by default (`/v1/chat/completions`, etc.) listening on `http://$HOST:$PORT`.

### 3. Smoke test

```bash
curl http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$SERVED_MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}"
```

---

## 5. Streaming Sessions (micro-turns)

The Interaction Model's real-time behavior comes from a **200ms micro-turn + persistent sequence** session model. To reproduce it, the client should:

1. Slice audio/video/text into `CHUNK_MS` (default 200ms) chunks;
2. Maintain a **session id** per conversation, tagging every 200ms chunk request with the same session id;
3. The server keeps appending chunks to the same GPU-resident sequence, avoiding a fresh prefill per chunk.

Relevant `.env` settings:

| Variable | Meaning | Default |
| --- | --- | --- |
| `CHUNK_MS` | Duration of each micro-turn (ms) | 200 |
| `AUDIO_SAMPLE_RATE` | Audio sample rate (Hz) | 16000 |
| `VIDEO_PATCH_SIZE` | Video patch edge length | 40 |
| `SESSION_MAX_SECONDS` | Max session duration (for reclaiming the persistent sequence) | 600 |

> Note: stock SGLang / vLLM expose a turn-based OpenAI API by default. A full "persistent sequence + 200ms streaming" setup requires a session-scheduling layer (session manager) on top of the backend that maps chunked requests onto the same KV sequence. This repo's `.env` already reserves the config knobs for that layer.

---

## 6. Environment Variables Cheat Sheet

See `.env.example` for the full list. Common ones:

| Variable | Description |
| --- | --- |
| `INFER_BACKEND` | Backend selection: `sglang` or `vllm` |
| `MODEL_PATH` | Weights path (local dir or HF repo) |
| `SERVED_MODEL_NAME` | Public-facing model name |
| `CUDA_VISIBLE_DEVICES` | Visible GPUs |
| `TP_SIZE` / `VLLM_TENSOR_PARALLEL_SIZE` | Tensor parallel size |
| `HOST` / `PORT` | Service listen address and port |
| `SGLANG_MEM_FRACTION_STATIC` / `VLLM_GPU_MEMORY_UTILIZATION` | GPU memory fraction |
| `CHUNK_MS` | micro-turn time slice |

---

## 7. References

- [Interaction Models: A Scalable Approach to Human-AI Collaboration — Thinking Machines Lab (official blog)](https://thinkingmachines.ai/blog/interaction-models/)
- [Mira Murati's Thinking Machines Lab Introduces Interaction Models — MarkTechPost](https://www.marktechpost.com/2026/05/13/mira-muratis-thinking-machines-lab-introduces-interaction-models-a-native-multimodal-architecture-for-real-time-human-ai-collaboration/)
- [Thinking Machines Lab Ships First Model With 200ms Real-Time Interaction — Unite.AI](https://www.unite.ai/thinking-machines-lab-ships-first-model-with-200ms-real-time-interaction/)
- [SGLang docs](https://docs.sglang.ai/)
- [vLLM docs](https://docs.vllm.ai/)
