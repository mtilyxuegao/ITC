#!/bin/bash
# Measure per-unit compute vs chunk_ms (eager). Runs inside the worker container.
cd /app
U=MiniCPMO45/modeling_minicpmo_unified.py
cp "$U" /tmp/U.bak
# make chunk extraction follow CHUNK_MS
sed -i 's/^            chunk_size = sample_rate$/            chunk_size = int(sample_rate * self.CHUNK_MS \/ 1000)/' "$U"
grep -n "chunk_size = int(sample_rate" "$U" | head -1

for X in 1000 500 250 100; do
  F=$((X+35))
  sed -i "s/self.CHUNK_MS = [0-9]*  # regular/self.CHUNK_MS = $X  # regular/; s/self.FIRST_CHUNK_MS = [0-9]*  # first/self.FIRST_CHUNK_MS = $F  # first/" "$U"
  echo "===================== chunk_ms=$X ====================="
  CUDA_VISIBLE_DEVICES=0 python benchmark.py --model-path /models/MiniCPM-o-4_5 \
    --video assets/samples/compile.mp4 --ref-audio assets/ref_audio/ref_en_dlc_1.wav --gpu-id 0 2>/dev/null \
    | grep -oE "unit=[0-9]+/[0-9]+ \| SPEAK[^|]*\| prefill:[^|]*\| generate:[^|]*\| unit_total=[0-9]+ms" \
    | tail -6
done
cp /tmp/U.bak "$U"   # restore
echo "DONE_SWEEP"
