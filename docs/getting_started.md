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
| `shardy` | the commit pinned in that XLA's `third_party/shardy/workspace.bzl` (`b6b99997`) | `shardy-mpmd-oss-mlir-import.patch` |

jax's `MODULE.bazel` can pin an older XLA than the fix; the build script
overrides it with the `xla` tree.

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

# The same model, weights and batches without the pipeline: one jax.jit
# program on all 8 GPUs, with replicated (dp) or sharded (fsdp) parameters.
XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS" python -m jaxtitan.trainer.train \
  --model=llama3.1-8b --parallel=dp \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100
```

Prepare the training data once with `python -m jaxtitan.data.text`, which
downloads WikiText-103 and writes Llama 3 token ids to `${JAXTITAN_ROOT}/data`.
The tokenizer comes from a sibling MaxText checkout (override with
`JAXTITAN_LLAMA3_TOKENIZER`).

## Results (2026-11-04, 8x MI355X)

Llama 3.1 8B with 65,536 tokens per step (32 sequences of 2,048 tokens), full
remat, XLA attention and AdamW on WikiText-103. All runs start from the same
weights and train on the same batches, so their step-0 losses match. Step
times are medians over steps 5-99. Per-step metrics are in `results/`.
A 30-step recheck on 2026-11-05, with jax `6b39d495e5` and XLA `d23c30982a`
and no XLA patch, gave the same pipeline result: 1.69 s per step and 181-182
GiB per GPU.

The pipeline runs 2 stages of 4 GPUs, data parallel inside each stage, with
1F1B over 8 microbatches of 4 sequences. `--reduce_grads` sets when a stage
sums gradients across its GPUs: once per step (the default) or in the backward
pass of every microbatch. The baselines run the same step without the
pipeline, as one `jax.jit` program on all 8 GPUs.

| | Pipeline, reduce per step | Pipeline, reduce per microbatch | No pipeline, DP | No pipeline, FSDP |
|---|---|---|---|---|
| Flags | `--parallel=mpmd` | `--parallel=mpmd --reduce_grads=microbatch` | `--parallel=dp` | `--parallel=fsdp` |
| Compile time | 113.7 s | 72.9 s | 128.6 s | 37.1 s |
| Step time | 1.70 s | 2.20 s | 0.83 s | 0.80 s |
| Throughput | 38.6k tokens/s | 29.8k tokens/s | 78.8k tokens/s | 81.5k tokens/s |
| MFU | 9.0% | 7.0% | 18.4% | 19.0% |
| Peak HBM per GPU | 181-182 GiB | 166-172 GiB | 127 GiB | 26 GiB |
| Loss at step 0, mean of steps 90-99 | 12.59, 6.71 | 12.59, 6.55 | 12.59, 6.85 | 12.59, 6.82 |
| Log | `llama3.1-8b_mpmd_pp2.jsonl` | `llama3.1-8b_mpmd_pp2_reduce_per_microbatch.jsonl` | `llama3.1-8b_dp8.jsonl` | `llama3.1-8b_fsdp8.jsonl` |

Reducing gradients once per step makes the pipeline 1.3x faster. Each GPU
trains its own copy of its stage's parameters, so microbatch gradients stay on
the GPU and the stage sums them once per step instead of 8 times, for 10-15 GiB
more memory per GPU.

The per-microbatch run reproduces the step time, throughput and memory of the
first run (`results/llama3.1-8b_pp2_first_run.jsonl`). Later losses differ
between runs because the pipeline clips gradients per stage while the
baselines clip their global norm, and because GPU runs are not bitwise
reproducible: the two per-microbatch runs end at 6.35 and 6.55.

On one node, the pipeline is still 2.0x slower than DP and 2.1x slower than
FSDP. Pipeline parallelism is not expected to win for an 8B model on 8 GPUs
with fast links; it pays off when a model or cluster outgrows data parallelism
and FSDP, for example across nodes. Two costs remain that the baselines don't
have:

- With 2 stages and 8 microbatches, the 1F1B schedule leaves each stage idle
  for at least 1/9 of the step.
- Each GPU processes 1 sequence at a time instead of 4, so it runs 4x as many
  matrix multiplications, each 4x smaller.
