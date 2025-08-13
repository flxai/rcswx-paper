#!/usr/bin/env bash
# Usage: scripts/einsearch-progress.sh /path/to/dir
# Recursively finds *.txt in dir and prints progress info for each

set -euo pipefail
(( $# >= 1 )) || { echo "Usage: $0 /path/to/dir" >&2; exit 2; }
rootdir=$1

preprocess() {
  sed -r 's/\x1b\[[0-9;]*[A-Za-z]//g; s/\r/\n/g' "$1"
}

for logfile in $(find "$rootdir" -type f -name '*.txt' | sort); do
  iterations=$(
    preprocess "$logfile" | grep -E '([0-9]+/[0-9]+).*(ETA|< *[0-9]{1,3}:[0-9]{2}:[0-9]{2})' \
      | tail -n1 | grep -Eo '[0-9]+/[0-9]+' || true
  )
  if [[ -z "${iterations:-}" ]]; then
    iterations=$(preprocess "$logfile" | grep -Eo '[0-9]+/[0-9]+' | tail -n1 || true)
  fi

  dataset=$(preprocess "$logfile" | grep -Eoi 'dataset[[:space:]=:]+[A-Za-z0-9_.+-]+' \
    | head -n1 | awk '{print $2}' || true)

  strategy=$(preprocess "$logfile" | grep -Eoi 'crossover[_ -]*strategy[[:space:]=:]+[A-Za-z0-9_.+-]+' \
    | tail -n1 | awk '{print $2}' || true)

  rate=$(preprocess "$logfile" | grep -Eoi 'crossover[_ -]*rate[[:space:]=:]+[0-9.]+$' \
    | tail -n1 | grep -Eo '[0-9.]+' || true)
  if [[ "${rate:-}" == "0" || "${rate:-}" == "0.0" ]]; then
    strategy="None"
  fi

  eta=$(
    preprocess "$logfile" \
      | grep -Eo '(ETA[^0-9]{0,10})?[< ]?[0-9]{1,3}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?[>]?' \
      | tail -n1 \
      | grep -Eo '[0-9]{1,3}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?' \
      || true
  )

  echo "$logfile"
  echo "  iterations: ${iterations}"
  echo "  dataset: ${dataset}"
  echo "  crossover_strategy: ${strategy}"
  echo "  crossover_rate: ${rate}"
  echo "  eta: ${eta}"
done

