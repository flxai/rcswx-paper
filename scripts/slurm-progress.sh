#!/usr/bin/env bash
# Usage: slurm_user_counts.sh [SINCE_YYYY-MM-DD]
set -euo pipefail
SINCE="${1:-$(date +%Y-%m-%d)}"

{
  echo "RUNNING QUEUED FINISHED FAILED TOTAL USER"

  awk '
    NR==FNR { # File1: from squeue
      r[$1]+=($2=="RUNNING"); q[$1]++; seen[$1]=1; next
    }
    { # File2: from sacct
      s=$2
      if (s ~ /COMPLETED/) fin[$1]++
      else if (s ~ /(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)/) fail[$1]++
      seen[$1]=1
    }
    END {
      for (u in seen) {
        running = r[u]+0; queued = q[u]+0
        finished = fin[u]+0; failed = fail[u]+0
        total = queued + finished + failed
        printf "%7d %6d %8d %6d %7d %s\n", running, queued, finished, failed, total, u
      }
    }
  ' <(squeue -h -o "%u %T") \
    <(sacct -a -X -S "$SINCE" --noheader -o User,State)
} | { read -r h; echo "$h"; sort -k2,2nr; } | column -t
