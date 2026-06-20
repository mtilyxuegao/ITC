#!/bin/bash
# DUET zombie-worker watchdog. Auto-recovers the stuck MiniCPM duplex worker that the gateway
# never releases on tab-close/refresh. CONSERVATIVE trigger so it never kills a legit session:
#   busy>=1 AND idle==0 AND queue_length>=1 (someone is BLOCKED waiting) AND held >THRESHOLD s,
#   sustained for STREAK polls; then scancel+resubmit serve_interaction. COOLDOWN prevents loops.
# Dormant when nobody is connecting (queue==0). Stop it: kill this script's PID (see duet-watchdog.log).
THRESHOLD=150; STREAK=3; POLL=20; COOLDOWN=420
EXCL_FIXED=liquid-gpu-001
last_restart=-9999; streak=0
echo "$(date +%H:%M:%S) watchdog started (pid $$): trigger busy>=1,idle=0,queue>=1,elapsed>${THRESHOLD}s x${STREAK}"
while true; do
  node=$(grep -oP 'MINICPM_NODE=\K\S+' /home/justin/duet-minicpm-endpoint.env 2>/dev/null)
  job=$(grep -oP 'MINICPM_JOB=\K\S+' /home/justin/duet-minicpm-endpoint.env 2>/dev/null)
  tnode=$(grep -oP 'SERVE_NODE=\K\S+' /home/justin/duet-endpoints.env 2>/dev/null)
  vals=$(curl -s --max-time 5 "http://$node:8006/status" 2>/dev/null | python3 -c "import sys,json
try:
 d=json.load(sys.stdin); rt=d.get('running_tasks',[]); el=max([t.get('elapsed_s',0) for t in rt],default=0)
 print(d.get('busy_workers',0),d.get('idle_workers',0),d.get('queue_length',0),int(el))
except: print('0 1 0 0')" 2>/dev/null)
  set -- $vals; busy=${1:-0}; idle=${2:-1}; q=${3:-0}; el=${4:-0}
  if [ "${busy:-0}" -ge 1 ] && [ "${idle:-1}" -eq 0 ] && [ "${q:-0}" -ge 1 ] && [ "${el:-0}" -gt $THRESHOLD ]; then
    streak=$((streak+1))
    echo "$(date +%H:%M:%S) WEDGE? job=$job busy=$busy idle=$idle queue=$q elapsed=${el}s streak=$streak/$STREAK"
  else
    streak=0
  fi
  if [ $streak -ge $STREAK ] && [ $((SECONDS - last_restart)) -gt $COOLDOWN ]; then
    echo "$(date +%H:%M:%S) !! AUTO-RECOVER: scancel $job + resubmit (exclude $EXCL_FIXED,$tnode)"
    scancel "$job" 2>/dev/null
    sbatch --exclude=${EXCL_FIXED},${tnode} /home/justin/ITC/duet/deploy/serve_interaction.sbatch 2>&1 | sed 's/^/   /'
    last_restart=$SECONDS; streak=0
  fi
  sleep $POLL
done
