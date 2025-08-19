#!/usr/bin/env bash
# scripts/slurm-progress.sh
# Usage: scripts/slurm-progress.sh [SINCE_YYYY-MM-DD]
set -euo pipefail

SINCE="${1:-$(date +%Y-%m-%d)}"
TODAY="$(date +%Y-%m-%d)"

# Cache inputs to avoid /dev/fd issues
sq="$(mktemp)"; sa="$(mktemp)"; trap 'rm -f "$sq" "$sa"' EXIT

squeue -h -o "%u %T" > "$sq"
if [[ "$SINCE" > "$TODAY" ]]; then
  : > "$sa"
else
  sacct -a -X -S "$SINCE" --noheader -o User,State%30 > "$sa" 2>/dev/null || :
fi

{
  echo "RUNNING QUEUED FINISHED FAILED OOM TOTAL USER"

  awk '
    NR==FNR {                      # squeue
      r[$1]+=($2=="RUNNING")
      q[$1]+=($2=="PENDING")       # queued == pending
      seen[$1]=1
      next
    }
    {                              # sacct
      s=$2
      if (s=="COMPLETED")                         fin[$1]++
      else if (s ~ /^OUT_OF_MEMORY/)              oom[$1]++
      else if (s ~ /(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)/) fail[$1]++
      seen[$1]=1
    }
    END {
      for (u in seen) {
        running = r[u]+0
        queued  = q[u]+0
        finished= fin[u]+0
        failed  = fail[u]+0
        oomc    = oom[u]+0
        total   = running + queued + finished + failed + oomc
        printf "%7d %6d %8d %6d %3d %7d %s\n", \
               running, queued, finished, failed, oomc, total, u
      }
    }
  ' "$sq" "$sa"
} | { read -r h; echo "$h"; sort -k2,2nr; } | column -t
