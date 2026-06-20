#!/usr/bin/env bash
#
# Start the Thinker (large reasoning model) on the H100 server's free GPUs (2,3).
# MiniCPM-o (the Talker reference) keeps GPU 0,1.
#
# Serves an OpenAI-compatible endpoint on :8000 that agent.py's ThinkerClient calls.
#
# Stable-config notes (learned the hard way on this box; Ubuntu 22.04, driver 570 /
# CUDA 12.8, vLLM 0.10.1.1 + torch 2.7.1+cu128 + transformers 4.57 + starlette 0.47):
#   --enforce-eager        : skip CUDA-graph capture; a TP=2 worker segfaulted ~25s
#                            after startup WITH graphs. Eager is rock-solid here.
#   NCCL_P2P_DISABLE=1     : rule out P2P issues between the two TP GPUs.
#   --reasoning-parser deepseek_r1 : QwQ emits <think> CoT; this splits it into
#                            `reasoning_content` so `content` is ONLY the conclusion
#                            (so the Talker never speaks the chain-of-thought).
#   Do NOT put a Starlette BaseHTTPMiddleware in front of vLLM — it breaks the
#   disconnect-based request abort the barge-in path relies on (vllm#10087).
#
# Usage:  bash start_thinker.sh
set -euo pipefail

MODEL="${THINKER_MODEL_PATH:-/home/ubuntu/models/QwQ-32B}"
SERVED_NAME="${SERVED_NAME:-itc-thinker}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-2,3}"
MAXLEN="${MAXLEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
VENV="${VENV:-$HOME/thinker-venv}"

echo "==> Thinker: $MODEL  (served as '$SERVED_NAME')  GPUs=$GPUS  TP=2  port=$PORT"

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAXLEN" \
  --enforce-eager \
  --reasoning-parser deepseek_r1 \
  --port "$PORT"
