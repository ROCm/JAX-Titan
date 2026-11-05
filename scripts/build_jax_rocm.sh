#!/usr/bin/env bash
# Builds the jax, jaxlib, jax-rocm-plugin and jax-rocm-pjrt wheels from source,
# following the "Building jax-ml/jax:main against openxla/xla:main" recipe on
# the ROCm/xla rocm-dev-infra branch. Run inside the TheRock container.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
JAXTITAN_ROOT=${JAXTITAN_ROOT:-$(dirname "${REPO}")/.jaxtitan}
JAX_SRC=${JAX_SRC:-${JAXTITAN_ROOT}/src/jax}
XLA_SRC=${XLA_SRC:-${JAXTITAN_ROOT}/src/xla}
# Must be the Shardy commit pinned by XLA, with the Shardy patches applied.
SHARDY_SRC=${SHARDY_SRC:-${JAXTITAN_ROOT}/src/shardy}
BAZEL_ROOT=${BAZEL_ROOT:-${JAXTITAN_ROOT}/bazel}
AMDGPU_TARGETS=${AMDGPU_TARGETS:-gfx950}
WHEELS=${WHEELS:-jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt}
# Leave headroom on shared machines.
BAZEL_CPUS=${BAZEL_CPUS:-160}
ROCM_DISTRO_URL=${ROCM_DISTRO_URL:-https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-10.0.0.tar.gz}
ROCM_DISTRO_HASH=${ROCM_DISTRO_HASH:-1c5e807875d26a2470ecc7323daa5b5b9009208a55c3290ac255a909cde15fc6}

cd "${JAX_SRC}"
mkdir -p dist

# ROCM_PATH must stay empty so Bazel uses the TheRock tarball.
python3 build/build.py build \
  --wheels="${WHEELS}" \
  --rocm_amdgpu_targets="${AMDGPU_TARGETS}" \
  --rocm_version=10 \
  --bazel_options=--override_module=xla="${XLA_SRC}" \
  --bazel_options=--override_repository=shardy="${SHARDY_SRC}" \
  --bazel_options=--config=rocm_clang_hermetic \
  --bazel_options=--repo_env=ROCM_PATH= \
  --bazel_options=--repo_env=ROCM_DISTRO_URL="${ROCM_DISTRO_URL}" \
  --bazel_options=--repo_env=ROCM_DISTRO_HASH="${ROCM_DISTRO_HASH}" \
  --bazel_options=--repo_env=TF_ROCM_AMDGPU_TARGETS="${AMDGPU_TARGETS}" \
  --bazel_options=--local_resources=cpu="${BAZEL_CPUS}" \
  --bazel_startup_options="--bazelrc=build/rocm/rocm.bazelrc" \
  --bazel_startup_options="--output_user_root=${BAZEL_ROOT}"

ls -l dist/
