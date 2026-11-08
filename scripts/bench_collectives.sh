#!/bin/bash
# Collective time and bus bandwidth by message size, for RCCL and MORI, into
# OUT_DIR (default $JAX_TITAN_ROOT/logs/collectives/<date>):
#
#   bash scripts/bench_collectives.sh [OUT_DIR]
#
# Run inside the JAX-Titan container with the GPUs free (scripts/gpus_free.sh on
# the host). It builds rccl-tests against TheRock's RCCL into
# $JAX_TITAN_ROOT/build/rccl-tests, from $JAX_TITAN_ROOT/src/rccl-tests (a copy of
# rocm-systems' projects/rccl-tests; RCCL_TESTS_SRC overrides). It then sweeps
# all_reduce, all_gather, reduce_scatter, alltoall and sendrecv from MIN_BYTES to
# MAX_BYTES on each of GPU_COUNTS GPUs of this node, with the algorithm,
# protocol and channel count RCCL picks at each size (-A 1), and the all-reduce
# again with each of ALGOS forced. MORI-EP's dispatch and combine benchmark
# runs too when MORI is installed (MORI_EP_BENCH, MORI_PERF_OUT).
# python -m jax_titan.planning.collectives --calibrate OUT_DIR turns the outputs
# into the table the planner reads.
set -euo pipefail

ROOT=${JAX_TITAN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/.jax_titan}
OUT=${1:-$ROOT/logs/collectives/$(date +%Y-%m-%d)}
SRC=${RCCL_TESTS_SRC:-$ROOT/src/rccl-tests}
BUILD=$ROOT/build/rccl-tests
MIN_BYTES=${MIN_BYTES:-1K}
MAX_BYTES=${MAX_BYTES:-4G}
GPU_COUNTS=${GPU_COUNTS:-"2 4 8"}
OPS=${OPS:-"all_reduce all_gather reduce_scatter alltoall sendrecv"}
ALGOS=${ALGOS:-"Ring Tree"}
TIMEOUT=${TIMEOUT:-900}

SITE=$(python -c 'import _rocm_sdk_core, os; print(os.path.dirname(os.path.dirname(_rocm_sdk_core.__file__)))')
CORE=$SITE/_rocm_sdk_core
DEVEL=$SITE/_rocm_sdk_devel
export LD_LIBRARY_PATH=$CORE/lib:$DEVEL/lib:${LD_LIBRARY_PATH:-}

if [ ! -x "$BUILD/all_reduce_perf" ]; then
  [ -f "$SRC/src/common.cu" ] || { echo "no rccl-tests source in $SRC" >&2; exit 1; }
  arch=${GPU_TARGETS:-$("$DEVEL/llvm/bin/amdgpu-arch" 2>/dev/null | sort -u | head -1)}
  make -C "$SRC/src" MPI=0 ROCM_PATH="$DEVEL" NCCL_HOME="$DEVEL" \
    HIPCC="$DEVEL/llvm/bin/amdclang++" GPU_TARGETS="${arch:-gfx950}" BUILDDIR="$BUILD" -j32
fi

mkdir -p "$OUT"
python -m jax_titan.observability.env "$OUT/env.json" || true
gpus=$(python -c 'import glob; print(sum(open(p).read().strip() != "0" for p in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/gpu_id")))')

run() {  # name, GPUs, binary, env...
  local name=$1 n=$2 bin=$3
  shift 3
  local devices
  devices=$(seq -s, 0 $((n - 1)))
  echo "== $name"
  # One thread per GPU, as JAX's runtime drives them: one thread for all of them adds
  # its launches to every collective (46 us instead of 22 us for 8 GPUs on MI355X).
  env HIP_VISIBLE_DEVICES="$devices" "$@" timeout -s KILL "$TIMEOUT" "$BUILD/$bin" \
    -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -t "$n" -g 1 -w 5 -n 20 -A 1 \
    > "$OUT/$name.out" 2>&1 || echo "   failed: see $OUT/$name.out"
}

for n in $GPU_COUNTS; do
  [ "$n" -le "$gpus" ] || continue
  for op in $OPS; do
    run "rccl_${op}_g${n}" "$n" "${op}_perf"
  done
  for algo in $ALGOS; do
    run "rccl_all_reduce_g${n}_${algo,,}" "$n" all_reduce_perf NCCL_ALGO="$algo"
  done
done

if python -c 'import mori' 2>/dev/null; then
  bench=${MORI_EP_BENCH:-$(python -c 'import mori, os; print(os.path.join(os.path.dirname(mori.__file__), "../../tests/python/ops/bench_dispatch_combine.py"))')}
  if [ -f "$bench" ]; then
    echo "== mori_ep"
    MORI_PERF_OUT="$OUT/mori_perf.jsonl" timeout -s KILL "$TIMEOUT" python "$bench" \
      > "$OUT/mori_ep.out" 2>&1 || echo "   failed: see $OUT/mori_ep.out"
  fi
else
  echo "MORI is not installed: no MORI-EP sweep (see docs/planning.md)"
fi
echo "wrote $OUT"
