#!/usr/bin/env bash
# Runs a grid of jobs for one experiment, in its own process group, on CPU.
#   - jobs file: one shell command per line (blank lines and lines starting with # are skipped),
#     written to results/<experiment>/logs/jobs/job_<n>.sh;
#   - each job's output goes to results/<experiment>/logs/job_<n>.log, the launcher's to launcher.log;
#   - CUDA_VISIBLE_DEVICES is empty unless GPU=1 is set in the environment;
#   - --window also starts scripts/janela_execucao.sh on the grid's process group.
# Usage: bash scripts/launch_grid.sh <experiment> <jobs_file> [-P <parallel jobs>] [--window]
set -eu
EXP=${1:?experiment name}
JOBS=${2:?jobs file}
shift 2
P=4
WINDOW=0
while [ $# -gt 0 ]; do
  case "$1" in
    -P) P=$2; shift 2 ;;
    --window) WINDOW=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
LOGS="results/$EXP/logs"
mkdir -p "$LOGS"
[ "${GPU:-0}" = 1 ] || export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}

rm -rf "$LOGS/jobs"
mkdir -p "$LOGS/jobs"
n=0
while IFS= read -r line; do
  trimmed=${line#"${line%%[![:space:]]*}"}
  case "$trimmed" in ''|\#*) continue ;; esac
  n=$((n + 1))
  printf '{ %s\n} > %s/job_%03d.log 2>&1\n' "$line" "$LOGS" "$n" > "$(printf '%s/jobs/job_%03d.sh' "$LOGS" "$n")"
done < "$JOBS"
setsid bash -c "ls '$LOGS'/jobs/job_*.sh | xargs -P $P -n 1 nice -n 19 bash" > "$LOGS/launcher.log" 2>&1 &
PGID=$!
echo "$EXP: $n jobs, $P in parallel, PGID $PGID, logs in $LOGS"
if [ "$WINDOW" = 1 ]; then
  nohup bash scripts/janela_execucao.sh "$PGID" "$EXP" "$LOGS/window.log" > /dev/null 2>&1 &
fi
