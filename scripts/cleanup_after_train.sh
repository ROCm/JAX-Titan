#!/usr/bin/env bash
# Frees GPUs and disk after a JAX-Titan job, touching only the current user's
# processes and files:
#   1. stops this user's JAX-Titan processes (SIGTERM, then SIGKILL),
#   2. waits until none of this user's processes hold a GPU,
#   3. removes stale RCCL shared-memory segments (/dev/shm/nccl-*) of this user,
#   4. removes the Orbax checkpoint directories named with --checkpoint.
# Prints the plan and changes nothing unless --yes is given. Works on the host
# and inside the container, since the container user has the host uid.
#
#   scripts/cleanup_after_train.sh
#   scripts/cleanup_after_train.sh --yes
#   scripts/cleanup_after_train.sh --yes --checkpoint /dev/shm/${USER}_ckpt_run1
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
JAXTITAN_ROOT=${JAXTITAN_ROOT:-$(dirname "${REPO}")/.jaxtitan}
GRACE=${GRACE:-30}
# Processes started by JAX-Titan entry points.
PATTERN=${PATTERN:-'jaxtitan\.(trainer\.train|observability)|examples/[a-z_]+\.py|tests/|pytest'}
# Checkpoints are only removed below these directories.
ALLOWED_ROOTS=("/dev/shm" "/tmp" "${JAXTITAN_ROOT}")

yes=0
checkpoints=()
while (( $# )); do
  case $1 in
    --yes) yes=1 ;;
    --checkpoint) checkpoints+=("$2"); shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

run() {
  if (( yes )); then "$@"; else echo "  would run: $*"; fi
}

gpu_pids() {
  rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[[:space:]]/{print $1}'
}

own_gpu_pids() {
  local pid
  for pid in $(gpu_pids); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
      echo "${pid}"
    fi
  done
}

echo "== 1. JAX-Titan processes of ${USER}"
mapfile -t procs < <(ps -u "${USER}" -o pid=,etime=,args= \
  | grep -E "${PATTERN}" | grep -vE "grep|cleanup_after_train" || true)
if (( ${#procs[@]} )); then
  printf '  %s\n' "${procs[@]}" | cut -c1-160
  pids=$(printf '%s\n' "${procs[@]}" | awk '{print $1}')
  run kill -TERM ${pids}
  if (( yes )); then
    for _ in $(seq "${GRACE}"); do
      alive=$(for p in ${pids}; do kill -0 "$p" 2>/dev/null && echo "$p"; done)
      [[ -z "${alive}" ]] && break
      sleep 1
    done
    if [[ -n "${alive:-}" ]]; then
      echo "  still running after ${GRACE}s, sending SIGKILL: ${alive//$'\n'/ }"
      kill -KILL ${alive} 2>/dev/null || true
    fi
  fi
else
  echo "  none"
fi

echo "== 2. GPUs held by ${USER}"
if (( yes )); then
  for _ in $(seq 60); do
    [[ -z "$(own_gpu_pids)" ]] && break
    sleep 1
  done
fi
held=$(own_gpu_pids)
if [[ -n "${held}" ]]; then
  echo "  still holding GPUs (not JAX-Titan entry points, or not yet exited):"
  ps -o pid=,etime=,stat=,args= -p "${held//$'\n'/,}" | cut -c1-160 | sed 's/^/  /'
  echo "  stop them yourself if they are done; zombies (stat Z) vanish once reaped."
else
  echo "  none"
fi

echo "== 3. Stale RCCL shared-memory segments of ${USER}"
mapfile -t segments < <(find /dev/shm -maxdepth 1 -user "${USER}" -name 'nccl-*' 2>/dev/null)
if (( ${#segments[@]} )) && [[ -n "${held}" ]]; then
  echo "  ${#segments[@]} segments, kept because ${USER} still holds GPUs"
elif (( ${#segments[@]} )); then
  echo "  ${#segments[@]} segments, $(du -ch "${segments[@]}" | tail -1 | cut -f1)"
  run rm -f -- "${segments[@]}"
else
  echo "  none"
fi

echo "== 4. Checkpoints"
for ckpt in "${checkpoints[@]}"; do
  if [[ ! -d "${ckpt}" ]]; then
    echo "  skip ${ckpt}: not a directory"; continue
  fi
  real=$(realpath "${ckpt}")
  allowed=0
  for root in "${ALLOWED_ROOTS[@]}"; do
    root=$(realpath -m "${root}")
    [[ "${real}" == "${root}"/* ]] && allowed=1
  done
  if (( ! allowed )); then
    echo "  skip ${real}: not below ${ALLOWED_ROOTS[*]}"; continue
  fi
  if [[ "$(stat -c %U "${real}")" != "${USER}" ]]; then
    echo "  skip ${real}: owned by $(stat -c %U "${real}")"; continue
  fi
  if ! find "${real}" -maxdepth 3 -name _CHECKPOINT_METADATA -print -quit | grep -q .; then
    echo "  skip ${real}: no Orbax checkpoint inside"; continue
  fi
  echo "  ${real}: $(du -sh "${real}" | cut -f1)"
  run rm -rf -- "${real}"
done
(( ${#checkpoints[@]} )) || echo "  none given"

echo "== GPU state"
if [[ -x "${REPO}/scripts/gpus_free.sh" ]]; then
  "${REPO}/scripts/gpus_free.sh" || true
fi
(( yes )) || echo "dry run: rerun with --yes to apply"
