#!/usr/bin/env bash
# Usage: slurm_user_counts.sh [SINCE_YYYY-MM-DD]
set -euo pipefail
SINCE="${1:-$(date +%Y-%m-%d)}"

{
  echo "RUNNING QUEUED FINISHED FAILED OOM TOTAL USER"

  awk '
    NR==FNR { # squeue
      r[$1]+=($2=="RUNNING")
      q[$1]++
      seen[$1]=1
      next
    }
    { # sacct
      s=$2
      if (s == "COMPLETED") fin[$1]++
      else if (s == "OUT_OF_MEMORY") oom[$1]++
      else if (s ~ /(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED)/) fail[$1]++
      seen[$1]=1
    }
    END {
      for (u in seen) {
        running = r[u]+0
        queued = q[u]+0
        finished = fin[u]+0
        failed = fail[u]+0
        oomc = oom[u]+0
        total = queued + finished + failed + oomc
        printf "%7d %6d %8d %6d %3d %7d %s\n", \
               running, queued, finished, failed, oomc, total, u
      }
    }
  ' <(squeue -h -o "%u %T") \
    <(sacct -a -X -S "$SINCE" --noheader -o User,State%30)
} | { read -r h; echo "$h"; sort -k2,2nr; } | column -t

