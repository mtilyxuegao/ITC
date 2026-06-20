#!/bin/bash
# DUET-Lite ASR service: faster-whisper large-v3 on one H100. The reliable READ seam.
#   sbatch duet/deploy/serve_asr.sbatch
#   -> endpoint in /home/justin/duet-asr-endpoint.env
#
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=duet-asr
#SBATCH -o /home/justin/duet-asr.%j.log

set -uo pipefail
PY=/home/justin/hermes-env/bin/python
NODE=$(hostname); JOB=${SLURM_JOB_ID:-local}
export PYTHONPATH=/home/justin/ITC
export ASR_PORT=8020 ASR_MODEL=large-v3 ASR_LANGUAGE=en

echo "[asr] node=$NODE job=$JOB"; nvidia-smi -L
# faster-whisper on GPU needs cuBLAS + cuDNN; make sure they're present in the env.
~/hermes-env/bin/python -c "import ctranslate2" 2>/dev/null || uv pip install -p ~/hermes-env ctranslate2 >/dev/null 2>&1
uv pip install -p ~/hermes-env nvidia-cublas-cu12 nvidia-cudnn-cu12 >/dev/null 2>&1 || true
# make the pip-installed cuDNN/cuBLAS libs loadable
export LD_LIBRARY_PATH="$($PY -c 'import os,nvidia.cudnn,nvidia.cublas; print(os.path.dirname(nvidia.cudnn.__file__)+"/lib:"+os.path.dirname(nvidia.cublas.__file__)+"/lib")' 2>/dev/null):${LD_LIBRARY_PATH:-}"

$PY -m duet.asr_server > /home/justin/asr.$JOB.log 2>&1 &
APID=$!
tries=0
until curl -sf "http://localhost:8020/health" >/dev/null 2>&1; do
  kill -0 $APID 2>/dev/null || { echo "[asr] DIED"; tail -40 /home/justin/asr.$JOB.log; exit 1; }
  tries=$((tries+1)); [ $tries -gt 120 ] && { echo "[asr] timeout"; tail -40 /home/justin/asr.$JOB.log; exit 1; }
  sleep 5
done
cat > /home/justin/duet-asr-endpoint.env <<EOF
ASR_URL=http://$NODE:8020
ASR_NODE=$NODE
ASR_JOB=$JOB
EOF
echo "[asr] ready on $NODE:8020 (model=large-v3)"; cat /home/justin/duet-asr-endpoint.env
wait $APID
