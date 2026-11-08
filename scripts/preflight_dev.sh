#!/usr/bin/env bash
# Checks a shared dev node before a job: how much disk space is left, the users
# taking the most disk, and which users are on the GPUs. Reads only. Run on the
# host, not inside a container, so it sees every user.
#
# Disk per user adds each entry directly under SCAN_DIRS (/home /tmp /var/tmp)
# to its owner, and each container's writable layer to its user: the one its
# name starts with (`${USER}_...`, as scripts/container.sh names it), else the
# one whose home directory it mounts. Home directories are not world-readable,
# so the scan runs under `sudo -n` when allowed; without it, other users' sizes
# are undercounts. Container sizes take about 25 s; --no-docker skips them.
#
# GPU users come from /sys/class/kfd/kfd/proc, which lists every process with a
# GPU context and its memory on each GPU. Processes in containers usually run
# as root; they are listed under the container's user.
#
# Exits 0 when no process holds a GPU, every GPU is idle (MAX_USE_PCT and
# MAX_VRAM_PCT, as in scripts/gpus_free.sh) and every filesystem has at least
# MIN_FREE_PCT percent free; 1 otherwise.
#
#   scripts/preflight_dev.sh
#   scripts/preflight_dev.sh --top 20 --no-docker
set -euo pipefail

TOP_K=${TOP_K:-10}
MIN_FREE_PCT=${MIN_FREE_PCT:-10}
MAX_USE_PCT=${MAX_USE_PCT:-5}
MAX_VRAM_PCT=${MAX_VRAM_PCT:-5}
JOBS=${JOBS:-32}
read -r -a SCAN_DIRS <<< "${SCAN_DIRS:-/home /tmp /var/tmp}"

sizes=1
while (( $# )); do
  case $1 in
    -k|--top)
      if (( $# < 2 )); then echo "$1 needs a value" >&2; exit 2; fi
      TOP_K=$2; shift ;;
    --no-docker) sizes=0 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ ! ${TOP_K} =~ ^[1-9][0-9]*$ ]]; then
  echo "--top needs a positive integer, got '${TOP_K}'" >&2
  exit 2
fi

# Bytes as df -h prints them, rounded up: 1536 -> 1.5K, 712e9 -> 664G.
HUMAN='function up(x) { return x == int(x) ? x : int(x) + 1 }
function human(b,  u, i) {
  split("B K M G T P", u, " "); i = 1
  while (b >= 1024 && i < 6) { b /= 1024; i++ }
  if (i > 1 && b < 10) return sprintf("%.1f%s", up(b * 10) / 10, u[i])
  return sprintf("%d%s", up(b), u[i]) }'
human() { awk -v b="$1" "${HUMAN}"' BEGIN { print human(b) }'; }

tmp=$(mktemp -d)
trap 'rm -rf "${tmp}"' EXIT
sudo=()
if sudo -n true 2>/dev/null; then sudo=(sudo -n); fi

declare -A is_user
for d in /home/*/; do
  [[ -d ${d} ]] || continue
  d=${d%/}
  is_user[${d##*/}]=1
done

# The user a container belongs to: the one its name starts with, else the
# first user named in the paths it mounts (/home/<user>/..., /dev/shm/<user>).
container_user() {
  local part parts
  if [[ -n ${is_user[${1%%_*}]:-} ]]; then
    echo "${1%%_*}"
    return
  fi
  IFS='/ ' read -r -a parts <<< "$2"
  for part in "${parts[@]}"; do
    if [[ -n ${part} && -n ${is_user[${part}]:-} ]]; then
      echo "${part}"
      return
    fi
  done
  echo "?"
}

declare -A cname cuser
docker_ok=0
if docker info >/dev/null 2>&1; then
  docker_ok=1
  while IFS=$'\t' read -r id name mounts; do
    cname[${id}]=${name#/}
    cuser[${id}]=$(container_user "${name#/}" "${mounts}")
  done < <(docker ps -aq --no-trunc | xargs -r docker inspect \
    -f '{{.Id}}{{"\t"}}{{.Name}}{{"\t"}}{{range .Mounts}}{{.Source}} {{end}}')
  if (( sizes )); then
    docker ps -a --no-trunc --size --format '{{.ID}}\t{{.State}}\t{{.Size}}' \
      > "${tmp}/sizes" 2>/dev/null &
  fi
fi

echo "preflight_dev on $(hostname) at $(date -u '+%F %H:%M UTC')"
if [[ -f /.dockerenv ]]; then
  echo "  inside a container: run on the host to see every user"
fi

echo "== Disk space left"
df_paths=(/ "${SCAN_DIRS[@]}" /dev/shm)
if (( docker_ok )); then
  df_paths+=("$(docker info -f '{{.DockerRootDir}}' 2>/dev/null)")
fi
df -B1 --output=target,fstype,size,used,avail "${df_paths[@]}" 2>/dev/null \
  | awk -v min="${MIN_FREE_PCT}" -v low="${tmp}/low" -v lowest="${tmp}/lowest" "${HUMAN}"'
    NR == 1 { printf "  %-16s %-6s %7s %7s %7s %6s\n", "MOUNT", "TYPE", "SIZE", "USED",
              "FREE", "FREE%"; next }
    seen[$1]++ { next }
    {
      # As df computes Use%: used / (used + available), rounded up.
      pct = $4 * 100 / ($4 + $5); free = 100 - int(pct) - (pct > int(pct))
      printf "  %-16s %-6s %7s %7s %7s %5d%%%s\n", $1, $2, human($3), human($4),
             human($5), free, (free < min ? "  low" : "")
      if (free < min) print $1 " " free "%" > low
      if (!n++ || free < best) { best = free; where = $1 }
    }
    END { if (n) print best "% (" where ")" > lowest }' || true

echo "== GPUs"
nodes=/sys/class/kfd/kfd/topology/nodes
gpu_id=() gpu_bdf=() gpu_busy=() gpu_used=() gpu_total=()
for n in $(ls "${nodes}" 2>/dev/null | sort -n); do
  id=$(<"${nodes}/${n}/gpu_id")
  [[ ${id} != 0 ]] || continue
  read -r minor loc dom < <(awk '$1 == "drm_render_minor" {m = $2}
    $1 == "location_id" {l = $2} $1 == "domain" {d = $2} END {print m, l, d}' \
    "${nodes}/${n}/properties")
  dev=/sys/class/drm/renderD${minor}/device
  gpu_id+=("${id}")
  gpu_bdf+=("$(printf '%04x:%02x:%02x.%x' "${dom:-0}" $(( loc >> 8 )) \
    $(( (loc >> 3) & 31 )) $(( loc & 7 )))")
  gpu_busy+=("$(cat "${dev}/gpu_busy_percent" 2>/dev/null || echo 0)")
  gpu_used+=("$(cat "${dev}/mem_info_vram_used" 2>/dev/null || echo 0)")
  gpu_total+=("$(cat "${dev}/mem_info_vram_total" 2>/dev/null || echo 0)")
done

declare -A vram puser petime pargs pcont powner
pids=()
for d in /sys/class/kfd/kfd/proc/[0-9]*; do
  [[ -d ${d} ]] || continue
  pid=${d##*/}
  read -r user etime < <(ps -o user:32=,etime= -p "${pid}" 2>/dev/null) || continue
  held=0
  for i in "${!gpu_id[@]}"; do
    if [[ -r ${d}/vram_${gpu_id[i]} ]]; then
      vram["${pid} ${i}"]=$(<"${d}/vram_${gpu_id[i]}")
      held=1
    fi
  done
  (( held )) || continue
  pids+=("${pid}")
  puser[${pid}]=${user}
  petime[${pid}]=${etime}
  pargs[${pid}]=$(ps -o args= -p "${pid}" 2>/dev/null || true)
  pcont[${pid}]=-
  powner[${pid}]=${user}
  if [[ $(cat "/proc/${pid}/cgroup" 2>/dev/null) =~ ([0-9a-f]{64}) ]]; then
    cid=${BASH_REMATCH[1]}
    pcont[${pid}]=${cname[${cid}]:-${cid:0:12}}
    if [[ ( ${user} == root || ${user} =~ ^[0-9]+$ ) && ${cuser[${cid}]:-?} != "?" ]]; then
      powner[${pid}]=${cuser[${cid}]}
    fi
  fi
done

busy=() idle=() gpu_users=()
if (( ${#gpu_id[@]} )); then
  printf '  %-3s %-12s %5s %17s  %s\n' GPU BDF BUSY VRAM STATE
fi
for i in "${!gpu_id[@]}"; do
  # Which user is holding memory on this GPU, largest first: "xxx, yyy, ... 219.0G".
  holders=$(for pid in "${pids[@]}"; do
      if [[ -n ${vram["${pid} ${i}"]:-} ]]; then
        printf '%s\t%s\n' "${vram["${pid} ${i}"]}" "${powner[${pid}]}"
      fi
    done | awk -F'\t' "${HUMAN}"' { b[$2] += $1 }
      END { for (u in b) printf "%s\t%s %s\n", b[u], u, human(b[u]) }' \
    | sort -rn | cut -f2 | paste -sd, - | sed 's/,/, /g')
  used_pct=$(( gpu_total[i] ? gpu_used[i] * 100 / gpu_total[i] : 0 ))
  if (( gpu_busy[i] > MAX_USE_PCT || used_pct > MAX_VRAM_PCT )); then
    state="in use${holders:+ by ${holders}}"
    busy+=("${i}")
  elif [[ -n ${holders} ]]; then
    state="no load; context held by ${holders}"
    idle+=("${i}")
  else
    state=idle
    idle+=("${i}")
  fi
  printf '  %-3s %-12s %4s%% %17s  %s\n' "${i}" "${gpu_bdf[i]}" "${gpu_busy[i]}" \
    "$(human "${gpu_used[i]}") / $(human "${gpu_total[i]}")" "${state}"
done
(( ${#gpu_id[@]} )) || echo "  no AMD GPUs (${nodes} is missing)"

echo "== GPU processes"
if (( ${#pids[@]} )); then
  printf '  %-10s %8s %12s %-8s %-32s %-18s %s\n' USER PID ELAPSED "RUNS AS" \
    CONTAINER "VRAM BY GPU" COMMAND
fi
for pid in "${pids[@]}"; do
  # Consecutive GPUs holding the same amount are merged: 0:219.0G 1-7:1.9G.
  per_gpu=$(for i in "${!gpu_id[@]}"; do
      if [[ -n ${vram["${pid} ${i}"]:-} ]]; then
        printf '%s\t%s\n' "${i}" "${vram["${pid} ${i}"]}"
      fi
    done | awk -F'\t' "${HUMAN}"'
      function span() { return (start == end ? start : start "-" end) ":" last }
      { h = human($2) }
      NR > 1 && h == last && $1 == end + 1 { end = $1; next }
      NR > 1 { out = out sep span(); sep = " " }
      { start = end = $1; last = h }
      END { if (NR) print out sep span() }')
  printf '  %-10s %8s %12s %-8s %-32s %-18s %s\n' "${powner[${pid}]}" "${pid}" \
    "${petime[${pid}]}" "${puser[${pid}]}" "${pcont[${pid}]}" "${per_gpu}" \
    "${pargs[${pid}]:0:70}"
  gpu_users+=("${powner[${pid}]}")
done
(( ${#pids[@]} )) || echo "  none"

echo "== Top ${TOP_K} users by disk"
echo "  FILES: entries under ${SCAN_DIRS[*]} by owner; CONTAINERS: writable layers"
if (( ! ${#sudo[@]} )); then
  unreadable=0
  for d in /home/*/; do
    [[ -r ${d} && -x ${d} ]] || unreadable=$(( unreadable + 1 ))
  done
  echo "  no sudo: ${unreadable} home directories are unreadable and count as near 0"
fi
echo "  scanning..." >&2
"${sudo[@]}" find "${SCAN_DIRS[@]}" -mindepth 1 -maxdepth 1 -printf '%u\t%p\0' \
  > "${tmp}/owners" 2>/dev/null || true
"${sudo[@]}" find "${SCAN_DIRS[@]}" -mindepth 1 -maxdepth 1 -print0 2>/dev/null \
  | "${sudo[@]}" xargs -0 -r -P "${JOBS}" -n 8 du -0 -sx --block-size=1 \
  > "${tmp}/du" 2>/dev/null || true

declare -A owner file_bytes cont_bytes cont_count
while IFS=$'\t' read -r -d '' user path; do
  owner[${path}]=${user}
done < "${tmp}/owners"
while IFS=$'\t' read -r -d '' bytes path; do
  user=${owner[${path}]:-?}
  file_bytes[${user}]=$(( ${file_bytes[${user}]:-0} + bytes ))
done < "${tmp}/du"

wait
: > "${tmp}/containers"
if [[ -s ${tmp}/sizes ]]; then
  # Docker prints decimal units: 1.77GB is 1.77e9 bytes.
  while IFS=$'\t' read -r id state bytes; do
    user=${cuser[${id}]:-?}
    cont_bytes[${user}]=$(( ${cont_bytes[${user}]:-0} + bytes ))
    cont_count[${user}]=$(( ${cont_count[${user}]:-0} + 1 ))
    printf '%s\t%s\t%s\t%s\n' "${bytes}" "${user}" "${state}" \
      "${cname[${id}]:-${id:0:12}}" >> "${tmp}/containers"
  done < <(awk -F'\t' 'BEGIN { f["B"] = 1; f["kB"] = 1e3; f["MB"] = 1e6
      f["GB"] = 1e9; f["TB"] = 1e12; f["PB"] = 1e15 }
    { split($3, s, " "); n = s[1] + 0; u = s[1]; sub(/^[0-9.]+/, "", u)
      printf "%s\t%s\t%.0f\n", $1, $2, n * f[u] }' "${tmp}/sizes")
fi

printf '%s\n' "${!file_bytes[@]}" "${!cont_bytes[@]}" | sort -u | while IFS= read -r user; do
  [[ -n ${user} ]] || continue
  f=${file_bytes[${user}]:-0}
  c=${cont_bytes[${user}]:-0}
  printf '%s\t%s\t%s\t%s\t%s\n' $(( f + c )) "${user}" "${f}" "${c}" \
    "${cont_count[${user}]:-0}"
done | sort -t$'\t' -k1,1nr | awk -F'\t' -v k="${TOP_K}" "${HUMAN}"'
  NR == 1 { printf "  %-12s %8s %8s %11s\n", "USER", "TOTAL", "FILES", "CONTAINERS" }
  NR <= k { printf "  %-12s %8s %8s %11s%s\n", ($2 == "?" ? "(unknown)" : $2), human($1),
            human($3), ($5 ? human($4) : "-"), ($5 ? " (" $5 ")" : "") }'

echo "== Top ${TOP_K} containers by writable layer"
if [[ -s ${tmp}/containers ]]; then
  sort -t$'\t' -k1,1nr "${tmp}/containers" | awk -F'\t' -v k="${TOP_K}" "${HUMAN}"'
    NR == 1 { printf "  %8s  %-12s %-8s %s\n", "SIZE", "USER", "STATE", "NAME" }
    NR <= k { printf "  %8s  %-12s %-8s %s\n", human($1), ($2 == "?" ? "(unknown)" : $2),
              $3, $4 }'
  echo "  not counted: images and build cache, shared by all (docker system df)"
elif (( ! docker_ok )); then
  echo "  docker is not available to ${USER}"
else
  echo "  skipped (--no-docker)"
fi

echo "== Summary"
if [[ -s ${tmp}/low ]]; then
  echo "  disk: below ${MIN_FREE_PCT}% free on $(paste -sd' ' "${tmp}/low")"
else
  echo "  disk: lowest free $(cat "${tmp}/lowest" 2>/dev/null || echo '?')"
fi
users=$(printf '%s\n' "${gpu_users[@]}" | sort -u | paste -sd' ' - | sed 's/ /, /g')
echo "  gpu users: ${users:-none}"
list() { local IFS=,; echo "$*"; }
if (( ${#busy[@]} )); then
  echo "  gpus in use: $(list "${busy[@]}")"
fi
if (( ${#idle[@]} )); then
  echo "  gpus without load: $(list "${idle[@]}") (HIP_VISIBLE_DEVICES=$(list "${idle[@]}"))"
fi
if (( ${#pids[@]} || ${#busy[@]} )) || [[ -s ${tmp}/low ]]; then
  exit 1
fi
