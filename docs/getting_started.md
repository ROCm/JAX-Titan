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
| `jax` | jax-ml/jax main (`6b39d495e5` was tested) | `jaxlib-ifrt-ir-on-pjrt.patch` |
| `xla` | openxla/xla main at or after `dfca2c9`, which has the ROCm IFRT IR fix ([#52290](https://github.com/openxla/xla/pull/52290); `d23c30982a` was tested) | none |
| `shardy` | the commit pinned in that XLA's `third_party/shardy/workspace.bzl` (`b6b99997`) | `shardy-mpmd-oss-mlir-import.patch`, `shardy-mpmd-parse-partitioning-options.patch` |

jax's `MODULE.bazel` can pin an older XLA than the fix; the build script
overrides it with the `xla` tree, and the Shardy that XLA pins with the
patched `shardy` tree.

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

# CPU tests: every schedule and layout against jax.jit, and checkpoint resume.
python -m pytest tests/

# Llama 3.1 8B, 2 stages x 4 MI355X, zero-bubble schedule, WikiText-103.
XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS" python -m jaxtitan.trainer.train \
  --model=llama3.1-8b --num_stages=2 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100

# The same with FSDP inside each stage, 17 and 15 layers per stage, and a
# checkpoint every 50 steps; rerunning it resumes from the latest checkpoint.
XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS" python -m jaxtitan.trainer.train \
  --model=llama3.1-8b --num_stages=2 --stage_fsdp --stage_layers=17,15 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100 \
  --checkpoint_dir=${JAXTITAN_ROOT}/checkpoints/llama8b --checkpoint_every=50

# The same model, weights and batches without the pipeline: one jax.jit
# program on all 8 GPUs, with replicated (dp) or sharded (fsdp) parameters.
XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS" python -m jaxtitan.trainer.train \
  --model=llama3.1-8b --parallel=dp \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100
```

[pipelines.md](pipelines.md) describes the schedules, the layouts inside a
stage (`--stage_fsdp`, `--tensor_parallel`, `--stage_layers`) and
checkpointing.

When a job is done, free the GPUs for the next user. On the host,
`scripts/cleanup_after_train.sh` shows what it would do: stop your JAX-Titan
processes, remove your stale RCCL segments in `/dev/shm`, and remove the
checkpoint directories you name with `--checkpoint`. Rerun it with `--yes` to
apply. It never touches other users' processes or files.

Prepare the training data once with `python -m jaxtitan.data.text`, which
downloads WikiText-103 and writes Llama 3 token ids to `${JAXTITAN_ROOT}/data`.
The tokenizer comes from a sibling MaxText checkout (override with
`JAXTITAN_LLAMA3_TOKENIZER`).

## Results

[pipelines.md](pipelines.md#results-2026-11-05-8x-mi355x) has the step time,
memory and bubble of every schedule and layout on 8x MI355X. For Llama 3.1
8B with 65,536 tokens per step, the 2-stage zero-bubble pipeline takes 1.43 s
per step with 17 and 15 layers per stage, 1F1B takes 1.68 s, and the same
step without a pipeline takes 0.83 s with data parallelism and 0.80 s with
FSDP.
