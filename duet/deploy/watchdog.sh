#!/bin/bash
# DUET zombie-worker watchdog v2. Auto-recovers a stuck MiniCPM duplex worker even with NO one
# queued. WEDGE = busy>=1 & idle==0 & held>THRESHOLD s & GPU util ~0% (a real active session keeps
# the GPU busy; a wedged session sits at 0%). Sustained STREAK polls, COOLDOWN between restarts.
THRESHOLD=120; STREAK=2; POLL=20; COOLDOWN=300; EXCL=liquid-gpu-001
last_restart=-9999; streak=0
echo "$(date +%H:%M:%S) watchdog v2 (pid $$): busy>=1,idle=0,held>${THRESHOLD}s,GPU~0% x${STREAK}"
while true; do
  node=$(grep -oP 'MINICPM_NODE=\K\S+' /home/justin/duet-minicpm-endpoint.env 2>/dev/null)
  job=$(grep -oP 'MINICPM_JOB=\K\S+' /home/justin/duet-minicpm-endpoint.env 2>/dev/null)
  tnode=$(grep -oP 'SERVE_NODE=\K\S+' /home/justin/duet-endpoints.env 2>/dev/null)
  vals=$(curl -s --max-time 5 "http://$node:8006/status" 2>/dev/null | python3 -c "import sys,json
try:
 d=json.load(sys.stdin); rt=d.get('running_tasks',[]); el=max([t.get('elapsed_s',0) for t in rt],default=0)
 print(d.get('busy_workers',0),d.get('idle_workers',0),int(el))
except: print('0 1 0')" 2>/dev/null)
  set -- $vals; busy=${1:-0}; idle=${2:-1}; el=${3:-0}
  wedge=0
  if [ "${busy:-0}" -ge 1 ] && [ "${idle:-1}" -eq 0 ] && [ "${el:-0}" -gt $THRESHOLD ]; then
    gpu=$(srun --jobid=$job --overlap --ntasks=1 bash -c 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1' 2>/dev/null | grep -vE '^srun:' | tr -dc '0-9')
    [ -z "$gpu" ] && gpu=99
    echo "$(date +%H:%M:%S) suspect: busy=$busy idle=$idle held=${el}s gpu=${gpu}% streak=$streak"
    [ "$gpu" -lt 5 ] && wedge=1
  fi
  if [ "$wedge" = 1 ]; then streak=$((streak+1)); else streak=0; fi
  if [ $streak -ge $STREAK ] && [ $((SECONDS - last_restart)) -gt $COOLDOWN ]; then
    echo "$(date +%H:%M:%S) !! AUTO-RECOVER: scancel $job + resubmit (exclude $EXCL,$tnode)"
    scancel "$job" 2>/dev/null
    sbatch --exclude=${EXCL},${tnode} /home/justin/ITC/duet/deploy/serve_interaction.sbatch 2>&1 | sed 's/^/   /'
    last_restart=$SECONDS; streak=0
  fi
  sleep $POLL
done
