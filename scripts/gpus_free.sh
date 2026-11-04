#!/usr/bin/env bash
# Exits 0 when no process holds a GPU and every GPU is idle; prints the reason
# otherwise. Run on the host, not inside a container, so it sees all users.
set -euo pipefail

MAX_USE_PCT=${MAX_USE_PCT:-5}
MAX_VRAM_PCT=${MAX_VRAM_PCT:-5}

pids=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[[:space:]]/{print $1}')
if [[ -n "${pids}" ]]; then
  echo "busy: GPU processes ${pids//$'\n'/ }"
  ps -o user=,pid=,etime=,comm= -p "${pids//$'\n'/,}" 2>/dev/null || true
  exit 1
fi

busy=$(rocm-smi --showuse --showmemuse 2>/dev/null | awk -v u="${MAX_USE_PCT}" -v m="${MAX_VRAM_PCT}" '
  /GPU use \(%\)/ {split($0, a, ":"); if (a[3] + 0 > u) print $1 " use " a[3] "%"}
  /VRAM%/ {split($0, a, ":"); if (a[3] + 0 > m) print $1 " vram " a[3] "%"}')
if [[ -n "${busy}" ]]; then
  echo "busy: ${busy//$'\n'/, }"
  exit 1
fi
echo "free"
