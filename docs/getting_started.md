# Getting started

JAX-Titan runs in a TheRock ROCm container with JAX built from source, because
the stock jaxlib cannot compile Shardy MPMD programs without Pathways. All
build trees, the venv, data and logs live in `JAXTITAN_ROOT`, which defaults to
`.jaxtitan` next to this repository.

## 1. Container

On the host:

```bash
scripts/container.sh     # starts ${USER}_therock-10.1_jaxtitan from
                         # ghcr.io/rocm/jax-base-ubu24.therock-10.1
docker exec -it -u ${USER} ${USER}_therock-10.1_jaxtitan bash
```

The container mounts the repository's parent directory at the same path and
runs as your uid, so files keep their host owner.

## 2. Sources

Create worktrees in `${JAXTITAN_ROOT}/src` and apply the patches from
`third_party/patches`:

| Tree | Revision | Patch |
|---|---|---|
| `jax` | jax-ml/jax main (`8e6230833e` was tested) | `jaxlib-ifrt-ir-on-pjrt.patch` |
| `xla` | the commit pinned in jax's `MODULE.bazel` (`c6c791ab`) | `xla-ifrt-ir-rocm-platform.patch` |
| `shardy` | the commit pinned in XLA's `third_party/shardy/workspace.bzl` (`8c601ca5`) | `shardy-mpmd-oss-mlir-import.patch` |

## 3. Build

Inside the container, build the `jax`, `jaxlib`, `jax-rocm-plugin` and
`jax-rocm-pjrt` wheels for gfx950:

```bash
scripts/build_jax_rocm.sh                 # about 25 minutes on 256 cores
WHEELS=jaxlib scripts/build_jax_rocm.sh   # jaxlib only, about 2 minutes
```

The script follows the ROCm/xla `rocm-dev-infra` README: hermetic Clang and
the TheRock 10.0 tarball for the build. Runs use the container's ROCm 10.1.

## 4. Python environment

```bash
python3 -m venv --system-site-packages ${JAXTITAN_ROOT}/venv
source ${JAXTITAN_ROOT}/venv/bin/activate
D=${JAXTITAN_ROOT}/src/jax/dist
pip install $D/jax-*.whl $D/jaxlib-*.whl
pip install --no-deps $D/jax_rocm10_plugin-*.whl $D/jax_rocm10_pjrt-*.whl
pip install optax tiktoken pyarrow jaxtyping absl-py immutabledict typing_extensions chex
pip install --no-deps -e ${JAXTITAN_ROOT}/src/shardy/shardy/integrations/python/jax/mpmd
```

`--system-site-packages` exposes the container's TheRock `rocm` wheels. In
every new shell, run `source scripts/env.sh`.

## 5. Runs

The node is shared, so check it is idle on the host first:
`scripts/gpus_free.sh`.

```bash
# One-time data preparation (CPU only).
python -m jaxtitan.data.text

# Two-stage MLP step on two 4-GPU meshes, checked against jax.jit.
python examples/mpmd_two_mesh_spike.py --platform=rocm

# Llama pipeline step (microbatches, 1F1B) against a single-mesh reference.
python examples/llama_mpmd_parity.py --platform=cpu

# Llama 3.1 8B, 2 stages x 4 MI355X, 1F1B, WikiText-103.
XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS" python -m jaxtitan.trainer.train \
  --model=llama3.1-8b --num_stages=2 --schedule=ONE_FWD_ONE_BWD \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100
```

Prepare the training data once with `python -m jaxtitan.data.text`, which
downloads WikiText-103 and writes Llama 3 token ids to `${JAXTITAN_ROOT}/data`.
The tokenizer comes from a sibling MaxText checkout (override with
`JAXTITAN_LLAMA3_TOKENIZER`).

## First results (2026-11-04, 8x MI355X)

Llama 3.1 8B with 2 pipeline stages of 4 GPUs, data parallel within each
stage, 8 microbatches of 4 x 2048 tokens (65,536 tokens per step), full
remat and XLA attention. Per-step metrics are in
`results/llama3.1-8b_pp2_first_run.jsonl`.

| Metric | Value |
|---|---|
| Compile time | 68.5 s |
| Step time (steps 5-99) | 2.20 s |
| Throughput | 29.8k tokens/s |
| MFU | 7.0% |
| Peak HBM per GPU | 166-172 GiB |
| Loss | 12.59 at step 0, 6.35 mean over steps 90-99 |
