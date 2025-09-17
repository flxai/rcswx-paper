#!/usr/bin/env bash
# KISS Slurm watchdog with de-dup & optional CANCELLED retry.
# - Submits one job per file via your queue script (no sbatch here).
# - Adopts already-active jobs by name to avoid duplicates.
# - Retries on states listed in RETRY_STATES.
# - Optional stop-sentinel: <jobfile>.stop prevents retry after CANCELLED*.

set -euo pipefail
shopt -s nullglob

# --- Paths ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd -P)}"
QUEUE_SCRIPT="${QUEUE_SCRIPT:-$BASE_DIR/scripts/slurm-queue.sh}"  # override if needed

# --- Settings (env-overridable) ---
POLL_SECS="${POLL_SECS:-300}"
RETRY_STATES="${RETRY_STATES:-OUT_OF_MEMORY,FAILED,NODE_FAIL,PREEMPTED,TIMEOUT}"  # add CANCELLED if desired

# --- Args ---
PATTERN="${1:-}"
if [[ -z "$PATTERN" ]]; then
  echo "Usage: $(basename "$0") '<glob-of-job-files>'" >&2
  exit 1
fi

# --- Preflight: locate queue script ---
if [[ ! -f "$QUEUE_SCRIPT" ]]; then
  if [[ -f "$SCRIPT_DIR/slurm-queue.sh" ]]; then
    QUEUE_SCRIPT="$SCRIPT_DIR/slurm-queue.sh"
  else
    echo "Queue script not found. Set QUEUE_SCRIPT or place slurm-queue.sh in $BASE_DIR/scripts/ or $SCRIPT_DIR/" >&2
    exit 1
  fi
fi

log(){ printf '[%(%F %T)T] %s\n' -1 "$*" >&2; }
mk_jobname(){ local b; b="$(basename "$1")"; printf '%s' "${b%.*}"; }

should_retry() {
  local state="$1" IFS=',' arr; read -ra arr <<< "$RETRY_STATES"
  for s in "${arr[@]}"; do [[ "$state" == "$s" ]] && return 0; done
  return 1
}

active_jobid_by_name() {
  local name="$1"
  squeue --me -h -o "%i|%j|%T" \
  | awk -F'|' -v n="$name" '$2==n && ($3=="PENDING"||$3=="CONFIGURING"||$3=="RUNNING"||$3=="COMPLETING"){print $1; exit}'
}

submit_via_queue() {
  local file="$1" out jid
  if ! out="$(bash "$QUEUE_SCRIPT" "$file" 2>&1)"; then
    log "Submission failed for $(basename "$file"): $out"; return 1
  fi
  jid="$(grep -Eo 'Submitted batch job[[:space:]]+[0-9]+' <<<"$out" | awk '{print $4}' | tail -n1 || true)"
  [[ -z "$jid" ]] && jid="$(grep -Eo '^[0-9]+' <<<"$out" | tail -n1 || true)"  # --parsable fallback
  [[ -n "$jid" ]] || { log "Could not parse JobID for $(basename "$file"). Output:\n$out"; return 1; }
  printf '%s' "$jid"
}

job_state() {
  local jid="$1" st
  st="$(sacct -j "$jid" -n -P --format=JobID,State | awk -F'|' -v id="$jid" '$1==id{print $2; f=1} END{if(!f) print ""}')"
  printf '%s' "$st"
}

# --- Collect files ---
mapfile -t FILES < <(compgen -G "$PATTERN" || true)
((${#FILES[@]})) || { echo "No files matched: $PATTERN" >&2; exit 1; }

log "Watchdog start | files=${#FILES[@]} | poll=${POLL_SECS}s | queue=$(realpath -m "$QUEUE_SCRIPT") | retry_states=${RETRY_STATES}"

# --- State ---
declare -A NAME=() CUR_JID=() FINAL=() RETRIES=()
for f in "${FILES[@]}"; do
  NAME["$f"]="$(mk_jobname "$f")"
  CUR_JID["$f"]=""
  FINAL["$f"]=""
  RETRIES["$f"]=0
done

total=${#FILES[@]}
finished=0

# --- Initial: adopt or submit ---
for f in "${FILES[@]}"; do
  n="${NAME[$f]}"
  if jid="$(active_jobid_by_name "$n")" && [[ -n "$jid" ]]; then
    CUR_JID["$f"]="$jid"
    log "Adopted active: $n -> jid=$jid"
  else
    if jid="$(submit_via_queue "$f")"; then
      CUR_JID["$f"]="$jid"
      log "Submitted: $n -> jid=$jid"
      sleep 0.1
    else
      FINAL["$f"]="SUBMIT_FAIL"; ((finished++))
    fi
  fi
done

# --- Loop ---
while (( finished < total )); do
  sleep "$POLL_SECS"
  for f in "${FILES[@]}"; do
    [[ -n "${FINAL[$f]}" ]] && continue
    n="${NAME[$f]}"; jid="${CUR_JID[$f]}"
    [[ -z "$jid" ]] && continue

    st="$(job_state "$jid")"
    [[ -z "$st" ]] && continue  # sacct lag

    case "$st" in
      PENDING|CONFIGURING|RUNNING|COMPLETING) : ;;
      COMPLETED)
        FINAL["$f"]="COMPLETED"; CUR_JID["$f"]=""
        ((finished++)); log "Completed: $n (jid=$jid)"
        ;;
      *)
        # Optional stop-sentinel for CANCELLED*: don't restart if <file>.stop exists.
        if [[ "$st" == CANCELLED* ]] && [[ -f "$f.stop" ]]; then
          FINAL["$f"]="$st"; CUR_JID["$f"]=""
          ((finished++)); log "Cancelled (stop sentinel): $n (jid=$jid)"
          continue
        fi
        if should_retry "$st"; then
          RETRIES["$f"]=$((RETRIES["$f"]+1))
          CUR_JID["$f"]=""
          # Avoid duplicates: check if another active with same name appeared meanwhile
          if new_active="$(active_jobid_by_name "$n")" && [[ -n "$new_active" ]]; then
            CUR_JID["$f"]="$new_active"
            log "Found active while retrying: $n -> jid=$new_active (prev=$jid, state=$st)"
          else
            # Cool-down for CANCELLED to avoid storms (small pause)
            [[ "$st" == CANCELLED* ]] && sleep 5
            if jid2="$(submit_via_queue "$f")"; then
              CUR_JID["$f"]="$jid2"
              log "Retry #${RETRIES[$f]}: $n (prev=$jid, state=$st) -> new jid=$jid2"
            else
              FINAL["$f"]="SUBMIT_FAIL"; ((finished++))
              log "Retry submit failed: $n (prev=$jid, state=$st)"
            fi
          fi
        else
          FINAL["$f"]="$st"; CUR_JID["$f"]=""
          ((finished++)); log "Terminal (no retry): $n (jid=$jid, state=$st)"
        fi
        ;;
    esac
  done
done

# --- Summary ---
ok=0; fail=0
for f in "${FILES[@]}"; do
  if [[ "${FINAL[$f]}" == "COMPLETED" ]]; then ((ok++)); else ((fail++)); fi
done
log "Done. Completed=$ok Failed/Other=$fail"
printf '%-40s  %-14s  %s\n' "FILE" "FINAL_STATE" "RETRIES"
for f in "${FILES[@]}"; do
  printf '%-40s  %-14s  %d\n' "$(basename "$f")" "${FINAL[$f]}" "${RETRIES[$f]}"
done
