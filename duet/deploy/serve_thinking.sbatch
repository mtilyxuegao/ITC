#!/bin/bash
# Stand up the DUET-Lite thinking layer: two MoE models served persistently on ONE
# 8xH100 node via vLLM (OpenAI-compatible). GPUs 0-3 host the two thinkers; 4-7 are
# left free for the MiniCPM-o duplex interaction layer.
#
#   sbatch duet/deploy/serve_thinking.sbatch
#   # then read /home/justin/duet-endpoints.env for the URLs to point ThinkingClient at
#
#SBATCH -p defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --job-name=duet-serve
#SBATCH -o /home/justin/duet-serve.%j.log

set -uo pipefail
# Use vLLM's precompiled Triton/torch sampler instead of FlashInfer, which would
# JIT-compile a kernel at startup and needs ninja+nvcc (absent on the compute node).
# Without this, profile_run dies: flashinfer_sample -> run_ninja -> FileNotFoundError.
export VLLM_USE_FLASHINFER_SAMPLER=0
VLLM=/home/justin/duet-env/bin/vllm
NODE=$(hostname)
JOB=${SLURM_JOB_ID:-local}
ENDPOINTS=/home/justin/duet-endpoints.env

echo "[serve] node=$NODE job=$JOB"
nvidia-smi -L

# --- Thinker A: LFM2.5-8B-A1B on GPU 0,1 (Liquid MoE, 8.3B total / ~1.5B active) ---
# Swapped from Qwen3.5-35B-A3B (3B active) to LFM2.5-8B-A1B (~1.5B active) for FASTER decode
# (speed > agentic depth, per the brief). Served as "qwen" on :8001 so the proxy/hermes config
# Qwen3.5-35B-A3B (MoE, 3B active) — research model, reverted from LFM2.5 per request.
# GDN linear-attn -> --gdn-prefill-backend triton; tool/reasoning parsers qwen3_coder/qwen3.
# Served as "qwen" on :8001.
CUDA_VISIBLE_DEVICES=0,1 "$VLLM" serve Qwen/Qwen3.5-35B-A3B \
  --served-model-name qwen \
  --tensor-parallel-size 2 --host 0.0.0.0 --port 8001 \
  --max-model-len 131072 \
  --gdn-prefill-backend triton \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  > /home/justin/vllm-think.$JOB.log 2>&1 &
THINK_PID=$!

# --- Thinker B: gemma-4-12B-it on GPU 2,3 (DENSE 12B, audio+vision+text) ---
# Swapped from the 26B-A4B MoE (which has NO audio tower) to 12B-it, which is encoder-free
# omni: it ingests image+audio directly (--limit-mm-per-prompt). Paves the way to feed user
# audio straight to the backend and drop whisper STT. Needs vllm[audio] deps (librosa+soundfile,
# installed into duet-env). Served as "gemma" on :8002 so the proxy is unchanged.
CUDA_VISIBLE_DEVICES=2,3 "$VLLM" serve google/gemma-4-12B-it \
  --served-model-name gemma \
  --tensor-parallel-size 2 --host 0.0.0.0 --port 8002 \
  --max-model-len 131072 \
  --limit-mm-per-prompt '{"image":4,"audio":1}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  > /home/justin/vllm-gemma.$JOB.log 2>&1 &
GEMMA_PID=$!

# --- wait for both /health, then publish endpoints ---
wait_health () {
  local port=$1 name=$2 tries=0
  until curl -sf "http://localhost:$port/health" >/dev/null 2>&1; do
    tries=$((tries+1))
    if [ $tries -gt 180 ]; then echo "[serve] $name FAILED to come up"; return 1; fi
    sleep 5
  done
  echo "[serve] $name healthy on :$port"
}

wait_health 8001 qwen
wait_health 8002 gemma

cat > "$ENDPOINTS" <<EOF
# DUET-Lite thinking-layer endpoints (job $JOB on $NODE)
THINK_URL=http://$NODE:8001
THINK_MODEL=qwen
GEMMA_URL=http://$NODE:8002
GEMMA_MODEL=gemma
SERVE_NODE=$NODE
SERVE_JOB=$JOB
EOF
echo "[serve] endpoints published to $ENDPOINTS:"
cat "$ENDPOINTS"
echo "[serve] both servers up. holding the allocation (scancel $JOB to stop)."

# keep the allocation alive as long as either server runs
wait $THINK_PID $GEMMA_PID
