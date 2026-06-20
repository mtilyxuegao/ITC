#!/bin/bash
# Stand up the DUET-Lite thinking layer: two MoE models served persistently on ONE
# 8xH100 node via vLLM (OpenAI-compatible). GPUs 0-3 host the two thinkers; 4-7 are
# left free for the MiniCPM-o duplex interaction layer.
#
#   sbatch duet/deploy/serve_thinking.sbatch
#   # then read /home/justin/duet-endpoints.env for the URLs to point ThinkingClient at
#
#SBATCH -p defq
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --job-name=duet-serve
#SBATCH -o /home/justin/duet-serve.%j.log

set -uo pipefail
VLLM=/home/justin/duet-env/bin/vllm
NODE=$(hostname)
JOB=${SLURM_JOB_ID:-local}
ENDPOINTS=/home/justin/duet-endpoints.env

echo "[serve] node=$NODE job=$JOB"
nvidia-smi -L

# --- Thinker A: Qwen3.5-35B-A3B on GPU 0,1 (bf16, TP=2) ---
CUDA_VISIBLE_DEVICES=0,1 "$VLLM" serve Qwen/Qwen3.5-35B-A3B \
  --served-model-name qwen \
  --tensor-parallel-size 2 --host 0.0.0.0 --port 8001 \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > /home/justin/vllm-qwen.$JOB.log 2>&1 &
QWEN_PID=$!

# --- Thinker B: gemma-4-26B-A4B-it on GPU 2,3 (bf16, TP=2) ---
CUDA_VISIBLE_DEVICES=2,3 "$VLLM" serve google/gemma-4-26B-A4B-it \
  --served-model-name gemma \
  --tensor-parallel-size 2 --host 0.0.0.0 --port 8002 \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
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
QWEN_URL=http://$NODE:8001
QWEN_MODEL=qwen
GEMMA_URL=http://$NODE:8002
GEMMA_MODEL=gemma
SERVE_NODE=$NODE
SERVE_JOB=$JOB
EOF
echo "[serve] endpoints published to $ENDPOINTS:"
cat "$ENDPOINTS"
echo "[serve] both servers up. holding the allocation (scancel $JOB to stop)."

# keep the allocation alive as long as either server runs
wait $QWEN_PID $GEMMA_PID
