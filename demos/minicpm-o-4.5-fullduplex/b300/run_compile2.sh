#!/bin/bash
set -e
apt-get update -q >/dev/null 2>&1
apt-get install -y -q gcc g++ python3-dev >/dev/null 2>&1
echo "ptxas sm_103a: $(/ptxas13 --help 2>/dev/null | grep -o sm_103a | head -1)"
export TRITON_PTXAS_PATH=/ptxas13
cd /app
CUDA_VISIBLE_DEVICES=0 CC=gcc TRITON_PTXAS_PATH=/ptxas13 TORCHINDUCTOR_CACHE_DIR=/app/torch_compile_cache \
  python benchmark.py --model-path /models/MiniCPM-o-4_5 \
  --video assets/samples/compile.mp4 \
  --ref-audio assets/ref_audio/ref_en_dlc_1.wav \
  --gpu-id 0 --compile
