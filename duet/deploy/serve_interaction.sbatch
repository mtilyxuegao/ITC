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
# torch.compile: DEFAULT ON (pre-warm wired below). MiniCPM compiles llm.model + tts.model;
# lazy compile is covered by startup warmup_compile() (duplex path) + the force_speak PRE-WARM
# below, so turn-1 is hot, not a JIT stall. Reuse the warm Inductor cache from the earlier
# compiled backend so this warmup is fast. Flip OFF for a fast cold start: COMPILE_FLAG= sbatch ...
# (single worker only — the 2-worker variant hung loading the 2nd backend; the watchdog covers HA.)
# DEFAULT OFF: lazy JIT triggers on first forward (>90s, two dynamic-shape paths) — can't be
# reliably pre-warmed, turn-1 stalls; ~150-200ms steady win not worth it. ON: COMPILE_FLAG=--compile sbatch
COMPILE_FLAG="${COMPILE_FLAG:-}"
export CC=gcc CXX=g++
export TORCHINDUCTOR_CACHE_DIR=/home/justin/.cache/torchinductor_minicpm0
$PY -m py_backend.server --host 0.0.0.0 --port 22500 --gpu-id 0 --model-path "$MODEL" $COMPILE_FLAG \
  > /home/justin/minicpm-backend.$JOB.log 2>&1 &
BPID=$!
echo "[minicpm] backend pid $BPID, loading model (COMPILE_FLAG='$COMPILE_FLAG')..."
tries=0
until curl -sf http://127.0.0.1:22500/health 2>/dev/null | grep -q '"ready"\|ready'; do
  if ! kill -0 $BPID 2>/dev/null; then
    echo "[minicpm] BACKEND DIED. last log:"; tail -40 /home/justin/minicpm-backend.$JOB.log; exit 1
  fi
  tries=$((tries+1)); [ $tries -gt 240 ] && { echo "[minicpm] backend health timeout"; exit 1; }
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

# --- PRE-WARM: trigger lazy torch.compile on the FORCE_SPEAK path before any user connects ---
# smoke_force_speak.py opens ws://localhost:22500/backend, sends one control.force_speak, closes
# its session (backend allows one active session, so it must finish before a browser connects).
# Best-effort: a failed pre-warm only means turn-1 pays the JIT, so we never abort the launch.
if [ -n "$COMPILE_FLAG" ]; then
  echo "[minicpm] pre-warming force_speak path (compile on)..."
  pw_t0=$SECONDS
  if timeout 240 $PY /home/justin/smoke_force_speak.py \
       > /home/justin/minicpm-prewarm.$JOB.log 2>&1; then
    echo "[minicpm] force_speak pre-warm OK ($((SECONDS - pw_t0))s) — turn-1 should be hot"
  else
    echo "[minicpm] force_speak pre-warm FAILED/timed out ($((SECONDS - pw_t0))s); turn-1 will JIT. last log:"
    tail -20 /home/justin/minicpm-prewarm.$JOB.log
  fi
fi

cat > "$ENDPOINT" <<EOF
# DUET-Lite interaction-layer (MiniCPM-o 4.5 duplex) endpoint — job $JOB on $NODE
MINICPM_NODE=$NODE
MINICPM_JOB=$JOB
MINICPM_GATEWAY=http://$NODE:8006
MINICPM_REALTIME_WS=ws://$NODE:8006/v1/realtime
# pipelines: ?mode=video (duplex voice+vision) | ?mode=audio (duplex audio) | ?mode=chat (turn-based, text-in)
EOF
echo "[minicpm] endpoints published to $ENDPOINT:"; cat "$ENDPOINT"

# --- DUET proxy + public tunnel, CO-LOCATED on this node ---------------------------
# proxy<->gateway is now localhost (removes the login<->compute hop from the voice path).
# Sibling services (thinking layer + ASR) live on other nodes but are off the voice path
# (async). Both proxy and ngrok are children of this job → scancel cleans them up, and a
# zombie-recovery restart brings the whole stack back fresh on the new node.
HERMES_PY=/home/justin/hermes-env/bin/python
[ -f /home/justin/duet-endpoints.env ]     && source /home/justin/duet-endpoints.env      # THINK_URL, GEMMA_URL
[ -f /home/justin/duet-asr-endpoint.env ]  && source /home/justin/duet-asr-endpoint.env    # ASR_URL
PX_THINK=${THINK_URL:-http://liquid-gpu-026:8001}
PX_THINK_MODEL=${THINK_MODEL:-qwen}
PX_AUX=${GEMMA_URL:-http://liquid-gpu-026:8002}
PX_ASR=${ASR_URL:-http://liquid-gpu-037:8020}
MINICPM_GATEWAY=http://127.0.0.1:8006 THINK_URL=$PX_THINK THINKER_MODEL=$PX_THINK_MODEL \
  AUX_URL=$PX_AUX AUX_MODEL=gemma ASR_URL=$PX_ASR HERMES_TOOLSETS=web WEB_SEARCH_BACKEND=ddgs PROXY_PORT=8010 \
  PYTHONPATH=/home/justin/ITC:/home/justin/hermes-agent \
  $HERMES_PY -m duet.proxy > /home/justin/duet-proxy.$JOB.log 2>&1 &
PXPID=$!
echo "[duet] proxy pid $PXPID (gateway=localhost:8006, think=$PX_THINK model=$PX_THINK_MODEL, aux=$PX_AUX, asr=$PX_ASR)"
ptries=0
until curl -sf http://127.0.0.1:8010/omni >/dev/null 2>&1; do
  kill -0 $PXPID 2>/dev/null || { echo "[duet] proxy DIED"; tail -30 /home/justin/duet-proxy.$JOB.log; break; }
  ptries=$((ptries+1)); [ $ptries -gt 60 ] && { echo "[duet] proxy health timeout"; break; }
  sleep 2
done
echo "[duet] proxy ready on :8010"

# public HTTPS tunnel on THIS node — fixed reserved subdomain (team ngrok wildcard).
/home/justin/ngrok http 8010 --url https://duet-omni.openclaw-claw-train.ngrok.app \
  --log=stdout --log-format=logfmt > /home/justin/ngrok-duet.$JOB.log 2>&1 &
NGPID=$!
echo "[duet] public URL → https://duet-omni.openclaw-claw-train.ngrok.app/omni"

echo "[minicpm] all tiers up. (scancel $JOB to stop)"
wait $BPID $WPID $GPID $PXPID $NGPID
