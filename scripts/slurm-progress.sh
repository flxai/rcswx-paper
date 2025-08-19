#!/usr/bin/env bash
# scripts/slurm-progress.sh
# Usage: scripts/slurm-progress.sh [SINCE_YYYY-MM-DD]
set -euo pipefail

SINCE="${1:-$(date +%Y-%m-%d)}"
TODAY="$(date +%Y-%m-%d)"

# Inputs
SQUEUE_INPUT=<(squeue -h -o "%u %T")
# sacct errors if start time is in the future; skip it then
if [[ "$SINCE" > "$TODAY" ]]; then
  SACCT_INPUT=/dev/null
else
  SACCT_INPUT=<(sacct -a -X -S "$SINCE" --noheader -o User,State%30)
fi

{
  echo "RUNNING QUEUED FINISHED FAILED OOM TOTAL USER"

  awk '
    NR==FNR { # squeue: user state
      r[$1] += ($2=="RUNNING")
      q[$1] += ($2=="PENDING")   # queue only pending
      seen[$1]=1
      next
    }
    { # sacct: user state
      s=$2
      if (s == "COMPLETED")                      fin[$1]++
      else if (s ~ /^OUT_OF_MEMORY/)             oom[$1]++
      else if (s ~ /(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)/) fail[$1]++
      seen[$1]=1
    }
    END {
      for (u in seen) {
        running  = r[u]+0
        queued   = q[u]+0
        finished = fin[u]+0
        failed   = fail[u]+0
        oomc     = oom[u]+0
        total    = running + queued + finished + failed + oomc
        printf "%7d %6d %8d %6d %3d %7d %s\n", \
               running, queued, finished, failed, oomc, total, u
      }
    }
  ' "$SQUEUE_INPUT" "$SACCT_INPUT"
} | { read -r h; echo "$h"; sort -k2,2nr; } | column -t

