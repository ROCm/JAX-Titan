# Source inside the TheRock container: `source scripts/env.sh`.
JAXTITAN_REPO=${JAXTITAN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export JAXTITAN_ROOT=${JAXTITAN_ROOT:-$(dirname "${JAXTITAN_REPO}")/.jaxtitan}

source "${JAXTITAN_ROOT}/venv/bin/activate"

# The ROCm plugin wheels do not bundle ROCm; use the container's TheRock libs.
_rocm_root=$(rocm-sdk path --root)
export LD_LIBRARY_PATH="${_rocm_root}/lib:${_rocm_root}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _rocm_root

export PYTHONPATH="${JAXTITAN_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_COMPILATION_CACHE_DIR="${JAXTITAN_ROOT}/cache/jax"

# From the ROCm/xla rocm-dev-infra scaling book (llama7b-jax-fundamentals).
export JAXTITAN_ROCM_XLA_FLAGS="--xla_gpu_autotune_level=4 \
--xla_gpu_enable_triton_gemm=true \
--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_all_gather_combine_threshold_bytes=8589934592 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 \
--xla_gpu_all_reduce_combine_threshold_bytes=8589934592 \
--xla_gpu_enable_command_buffer="
