#!/usr/bin/env bash
# Builds JAX-AITER from the third_party/jax-aiter submodule against the jaxlib
# in the active venv and installs it there. Run inside the container after
# `source scripts/env.sh`, with the submodule and its AITER fetched:
#
#   git submodule update --init --recursive third_party/jax-aiter
#
# The AITER JIT libraries behind flash attention and RMSNorm take hours to
# compile; ci/fetch_jit_libs.sh downloads the prebuilt ones that match the
# pinned AITER commit instead. The FFI shims compile in about a minute.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/jax-aiter"
export JA_ROOT_DIR=${PWD} GPU_ARCHS=${GPU_ARCHS:-gfx950} AITER_SYMBOL_VISIBLE=1

# AITER's GEMM config generator imports pandas; keep numpy and jax as they are.
python -m pip install -q pandas "numpy==$(python -c 'import numpy; print(numpy.__version__)')" \
  "jax==$(python -c 'import jax; print(jax.__version__)')"

bash ci/fetch_jit_libs.sh
ls build/aiter_build/*.so >/dev/null
make -j16
make ja_mods -j16
JA_WHEEL_VARIANT=full python -m pip install --no-deps --no-build-isolation .
python -c "import jax_aiter.mha, jax_aiter.gemm, jax_aiter.rmsnorm; print('JAX-AITER installed')"
