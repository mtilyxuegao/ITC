#!/bin/bash
# Stand up the DUET-Lite INTERACTION layer: MiniCPM-o 4.5 native full-duplex demo,
# run WITHOUT Docker (the cluster has Slurm+enroot, no user Docker). Three Python
# processes on ONE GPU: backend (loads model) -> worker (forwarder) -> gateway.
#
#   sbatch duet/deploy/serve_interaction.sbatch
#   # then read /home/justin/duet-minicpm-endpoint.env for the realtime WS URL
#
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=duet-minicpm
#SBATCH -o /home/justin/duet-minicpm.%j.log

set -uo pipefail
REPO=/home/justin/MiniCPM-o-Demo
PY=/home/justin/minicpm-env/bin/python
MODEL=openbmb/MiniCPM-o-4_5
NODE=$(hostname)
JOB=${SLURM_JOB_ID:-local}
ENDPOINT=/home/justin/duet-minicpm-endpoint.env
cd "$REPO"
export PYTHONPATH=.

echo "[minicpm] node=$NODE job=$JOB"
nvidia-smi -L

# --- Tier 3: backend (the only GPU process; Slurm pins the 1 allocated GPU) ---
$PY -m py_backend.server --host 0.0.0.0 --port 22500 --gpu-id 0 --model-path "$MODEL" \
  > /home/justin/minicpm-backend.$JOB.log 2>&1 &
BPID=$!
echo "[minicpm] backend pid $BPID, loading model (may take a few minutes)..."
tries=0
until curl -sf http://127.0.0.1:22500/health 2>/dev/null | grep -q '"ready"\|ready'; do
  if ! kill -0 $BPID 2>/dev/null; then
    echo "[minicpm] BACKEND DIED. last log:"; tail -40 /home/justin/minicpm-backend.$JOB.log; exit 1
  fi
  tries=$((tries+1)); [ $tries -gt 120 ] && { echo "[minicpm] backend health timeout"; exit 1; }
  sleep 5
done
echo "[minicpm] backend ready on :22500"

# --- Tier 2: worker (pure forwarder, no GPU) ---
$PY worker.py --host 0.0.0.0 --port 22400 --backend-server-url http://127.0.0.1:22500 \
  > /home/justin/minicpm-worker.$JOB.log 2>&1 &
WPID=$!
until curl -sf http://127.0.0.1:22400/health >/dev/null 2>&1; do
  kill -0 $WPID 2>/dev/null || { echo "[minicpm] worker died"; tail -30 /home/justin/minicpm-worker.$JOB.log; exit 1; }
  sleep 2
done
echo "[minicpm] worker ready on :22400"

# --- Tier 1: gateway (HTTP; raw WS clients/H0 spike need no TLS) ---
$PY gateway.py --host 0.0.0.0 --port 8006 --http --workers localhost:22400 \
  > /home/justin/minicpm-gateway.$JOB.log 2>&1 &
GPID=$!
until curl -sf http://127.0.0.1:8006/health >/dev/null 2>&1; do
  kill -0 $GPID 2>/dev/null || { echo "[minicpm] gateway died"; tail -30 /home/justin/minicpm-gateway.$JOB.log; exit 1; }
  sleep 2
done
echo "[minicpm] gateway ready on :8006"

cat > "$ENDPOINT" <<EOF
# DUET-Lite interaction-layer (MiniCPM-o 4.5 duplex) endpoint — job $JOB on $NODE
MINICPM_NODE=$NODE
MINICPM_JOB=$JOB
MINICPM_GATEWAY=http://$NODE:8006
MINICPM_REALTIME_WS=ws://$NODE:8006/v1/realtime
# pipelines: ?mode=video (duplex voice+vision) | ?mode=audio (duplex audio) | ?mode=chat (turn-based, text-in)
EOF
echo "[minicpm] endpoints published to $ENDPOINT:"; cat "$ENDPOINT"
echo "[minicpm] all tiers up. (scancel $JOB to stop)"
wait $BPID $WPID $GPID
