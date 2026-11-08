# Getting started

JAX-Titan runs in a TheRock ROCm container with JAX built from source, because
the stock jaxlib cannot compile Shardy MPMD programs without Pathways. All
build trees, the venv, data and logs live in `JAX_TITAN_ROOT`, which defaults to
`.jax_titan` next to this repository.

## 1. Container

On the host:

```bash
scripts/container.sh     # starts ${USER}_therock-10.1_jax_titan from
                         # ghcr.io/rocm/jax-base-ubu24.therock-10.1
docker exec -it -u ${USER} ${USER}_therock-10.1_jax_titan bash
```

The container mounts the repository's parent directory at the same path and
runs as your uid, so files keep their host owner.

## 2. Sources

Create worktrees in `${JAX_TITAN_ROOT}/src` and apply the patches from
`third_party/patches`:

| Tree | Revision | Patch |
|---|---|---|
| `jax` | jax-ml/jax main (`6b39d495e5` was tested) | `jaxlib-ifrt-ir-on-pjrt.patch` |
| `xla` | openxla/xla main at or after `dfca2c9`, which has the ROCm IFRT IR fix ([#52290](https://github.com/openxla/xla/pull/52290); `d23c30982a` was tested) | none |
| `shardy` | the commit pinned in that XLA's `third_party/shardy/workspace.bzl` (`b6b99997`) | `shardy-mpmd-oss-mlir-import.patch`, `shardy-mpmd-parse-partitioning-options.patch` |

jax's `MODULE.bazel` can pin an older XLA than the fix; the build script
overrides it with the `xla` tree, and the Shardy that XLA pins with the
patched `shardy` tree.

The operator providers JAX-AITER and JAX-Triton, and the Gluon GEMMs of
gfx950-gluon-tutorials, are git submodules in `third_party/`, pinned to tested
commits. JAX-AITER has AITER as its own submodule, which the `triton` provider
takes AITER's Triton kernels from:

```bash
git submodule update --init third_party/jax-triton third_party/gfx950-gluon-tutorials
git submodule update --init --recursive third_party/jax-aiter   # with AITER
```

Transformer Engine is not built from source: ROCm/maxtext publishes nightly
ROCm TransformerEngine wheels for ROCm 10.1, Python 3.12 and MI355X on its
[releases page](https://github.com/ROCm/maxtext/releases); step 4 installs
one.

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
python3 -m venv --system-site-packages ${JAX_TITAN_ROOT}/venv
source ${JAX_TITAN_ROOT}/venv/bin/activate
D=${JAX_TITAN_ROOT}/src/jax/dist
pip install $D/jax-*.whl $D/jaxlib-*.whl
pip install --no-deps $D/jax_rocm10_plugin-*.whl $D/jax_rocm10_pjrt-*.whl
pip install optax tiktoken pyarrow jaxtyping absl-py immutabledict typing_extensions chex
pip install --no-deps -e ${JAX_TITAN_ROOT}/src/shardy/shardy/integrations/python/jax/mpmd
# Optional, for TE operators: the ROCm/maxtext nightly TE wheel of 2026-11-02
# (TE 199f548960fa). It pins jax==0.12.0; --no-deps keeps the jax built above.
pip install --no-deps "https://github.com/ROCm/maxtext/releases/download/te-rocm-wheels-2026-11-02-199f548960fa/transformer_engine-2.20.0.dev0%2B199f54896-1.mi355-cp312-cp312-linux_x86_64.whl"
# Optional, for JAX-AITER operators: build the submodule against this jaxlib
# (prebuilt AITER JIT libraries, about two minutes).
git submodule update --init --recursive third_party/jax-aiter
scripts/build_jax_aiter.sh
# Optional, for the Triton and Gluon operators: Triton from PyPI, which has the
# AMD backend and Gluon, and JAX-Triton from its submodule.
pip install "triton==3.9.0"
pip install --no-deps -e third_party/jax-triton
```

`--system-site-packages` exposes the container's TheRock `rocm` wheels. In
every new shell, run `source scripts/env.sh`.

For MaxText models, install the `maxtext` extra from the repository root:
MaxText 0.2.5 and the packages its model code imports. Naming the installed
jax and jaxlib keeps pip from replacing the ones built above.

```bash
pip install -e ".[maxtext]" "jax==$(python -c 'import jax; print(jax.__version__)')" \
  "jaxlib==$(python -c 'import jaxlib; print(jaxlib.__version__)')"
```

For MaxDiffusion's FLUX.1, install MaxDiffusion from a checkout of the pinned
commit with the `maxdiffusion-*` patches from `third_party/patches`, without
its own requirements (PyTorch, TensorFlow and libtpu), then the
`maxdiffusion` extra:

```bash
M=${JAX_TITAN_ROOT}/src/maxdiffusion
git clone https://github.com/AI-Hypercomputer/maxdiffusion $M
git -C $M checkout 821afc364ac1889c1039112843f4f7a17c1bef7b
git -C $M apply $PWD/third_party/patches/maxdiffusion-*.patch
pip install --no-deps -e $M
pip install -e ".[maxdiffusion]" "jax==$(python -c 'import jax; print(jax.__version__)')" \
  "jaxlib==$(python -c 'import jaxlib; print(jaxlib.__version__)')"
```

## 5. Runs

The node is shared, so check it is idle on the host first:
`scripts/gpus_free.sh`.

```bash
# One-time data preparation (CPU only).
python -m jax_titan.data.text

# Two-stage MLP step on two 4-GPU meshes, checked against jax.jit.
python examples/mpmd_two_mesh_spike.py --platform=rocm

# Llama pipeline step (microbatches, 1F1B) against a single-mesh reference.
python examples/llama_mpmd_parity.py --platform=cpu

# MaxText Llama 3.1 stages against MaxText's model, JAX-Titan's Llama and a
# single-mesh reference (needs the maxtext extra).
python examples/maxtext_llama_parity.py --platform=cpu

# MaxDiffusion FLUX.1-dev stages against MaxDiffusion's transformer and a
# single-mesh reference (needs MaxDiffusion, see above).
python examples/maxdiffusion_flux_parity.py --platform=cpu

# CPU tests: every schedule and layout against jax.jit, and checkpoint resume.
python -m pytest tests/

# Operator providers against native JAX on MI355X: outputs and gradients, on one
# GPU, under vmap, sharded, inside shard_map and inside an MPMD step, plus
# timings. Run TE and JAX-AITER in separate processes (see docs/pipelines.md).
python -m jax_titan.ops.conformance --providers=te,te_mxfp8,te_fp8
python -m jax_titan.ops.conformance --providers=aiter --modes=bench
python -m jax_titan.ops.conformance --providers=triton,gluon --modes=single,sharded,manual,mpmd,bench

# Llama 3.1 8B, 2 stages x 4 MI355X, zero-bubble schedule, WikiText-103.
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=llama3.1-8b --num_stages=2 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100

# The same with FSDP inside each stage, 17 and 15 layers per stage, and a
# checkpoint every 50 steps; rerunning it resumes from the latest checkpoint.
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=llama3.1-8b --num_stages=2 --stage_fsdp --stage_layers=17,15 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100 \
  --checkpoint_dir=${JAX_TITAN_ROOT}/checkpoints/llama8b --checkpoint_every=50

# The same model, weights and batches without the pipeline: one jax.jit
# program on all 8 GPUs, with replicated (dp) or sharded (fsdp) parameters.
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=llama3.1-8b --parallel=dp \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100

# MaxText's Llama 3.1 8B, built from MaxText's layers (the maxtext extra), on
# the same data and pipeline. Gradients are reduced per microbatch.
# --ops=attention=te or aiter routes MaxText's attention through jax_titan.ops.
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=maxtext:llama3.1-8b --num_stages=2 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100

# From pretrained weights: convert Hugging Face's Llama 3.1 8B once with
# MaxText's converter (needs a Hugging Face token with access to the model, or
# --hf_model_path to a local copy), then start from the converted parameters.
HF_AUTH_TOKEN=$HF_TOKEN python -m maxtext.checkpoint_conversion.to_maxtext \
  ${JAX_TITAN_ROOT}/src/maxtext/src/maxtext/configs/base.yml model_name=llama3.1-8b \
  base_output_directory=${JAX_TITAN_ROOT}/checkpoints/llama3.1-8b-hf hardware=cpu \
  skip_jax_distributed_system=True scan_layers=True --lazy_load_tensors=True
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=maxtext:llama3.1-8b --num_stages=2 --ops=attention=te \
  --init_from=${JAX_TITAN_ROOT}/checkpoints/llama3.1-8b-hf/0/items --steps=100

# From pretrained weights: convert Hugging Face's Llama 3.1 8B once with
# MaxText's converter (needs a Hugging Face token with access to the model, or
# --hf_model_path to a local copy), then start from the converted parameters.
HF_AUTH_TOKEN=$HF_TOKEN python -m maxtext.checkpoint_conversion.to_maxtext \
  ${JAX_TITAN_ROOT}/src/maxtext/src/maxtext/configs/base.yml model_name=llama3.1-8b \
  base_output_directory=${JAX_TITAN_ROOT}/checkpoints/llama3.1-8b-hf hardware=cpu \
  skip_jax_distributed_system=True scan_layers=True --lazy_load_tensors=True
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=maxtext:llama3.1-8b --num_stages=2 --ops=attention=te \
  --init_from=${JAX_TITAN_ROOT}/checkpoints/llama3.1-8b-hf/0/items --steps=100

# Its no-pipeline reference for Gate 0: the same stages, weights and batches as
# one data-parallel jax.jit program; then compare the first 10 steps' losses.
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=maxtext:llama3.1-8b --parallel=dp \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=10 --run_dir=runs/mt_dp

# Any run with --run_dir records the scaling book's training metrics (step
# records and timeline, env.json, GPU power, validation with --eval_every) and
# writes results.json; docs/metrics.md lists them and the tools that read them.
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=maxtext:llama3.1-8b --num_stages=2 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=10 --run_dir=runs/mt_pp2
python -m jax_titan.observability.summarize --gate0 runs/mt_dp runs/mt_pp2 \
  --gate0_steps 10 --gate0_rtol 2e-3

# MaxDiffusion's FLUX.1-dev (12B) on synthetic 1024x1024 batches: noised
# random latents and text embeddings, with the flow-matching loss. With 2
# stages, the first holds 7.9B parameters and runs out of memory.
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=maxdiffusion:flux-dev --num_stages=4 \
  --microbatches=8 --microbatch_size=4 --steps=20 --lr=1e-5
```

`--model_overrides` takes a JSON object of MaxText config or FLUX.1
transformer settings, for example
`'{"base_num_decoder_layers": 4}'` for a 4-layer MaxText model.

`--ops` picks operator providers per op, with native JAX as the fallback:
`--ops=attention=te` runs TE's fused attention, and
`--ops=attention=te,linear=te_fp8,swiglu=te_fp8` adds FP8 GEMMs, and
`--ops=rms_norm=triton,rope=triton,linear=gluon` runs AITER's Triton RMSNorm and
RoPE with Gluon GEMMs. Providers whose kernels cannot run under `jax.vmap`
(TE's GEMMs, JAX-AITER, and the `triton` and `gluon` providers) need
`--per_device=shard_map` when gradients are reduced per step; without it they
fall back to native, and the trainer prints what each op ran on after
compiling.

[pipelines.md](pipelines.md) describes the schedules, the layouts inside a
stage (`--stage_fsdp`, `--tensor_parallel`, `--stage_layers`) and
checkpointing.

Before a large run, the budget planner estimates its time, GPU-hours, energy
and memory, on one GPU type or several side by side (CPU only;
[planning.md](planning.md)):

```bash
python -m jax_titan.planning.budget --model=maxtext:llama3.1-70b --tokens=15e12 \
  --gpu=MI300X,MI325X,MI355X --gpus=2048 --tp=8 --fsdp
python -m jax_titan.planning.budget --model=maxtext:llama3.1-8b --tokens=2e12 \
  --gpu=MI325X --gpus=256 --seq_len=2048 --from_run=runs/mt_pp2
```

When a job is done, free the GPUs for the next user. On the host,
`scripts/cleanup_after_train.sh` shows what it would do: stop your JAX-Titan
processes, remove your stale RCCL segments in `/dev/shm`, and remove the
checkpoint directories you name with `--checkpoint`. Rerun it with `--yes` to
apply. It never touches other users' processes or files.

Prepare the training data once with `python -m jax_titan.data.text`, which
downloads WikiText-103 and writes Llama 3 token ids to `${JAX_TITAN_ROOT}/data`.
The tokenizer comes from a sibling MaxText checkout (override with
`JAX_TITAN_LLAMA3_TOKENIZER`).

## Results

[recipes/maxtext-llama3.1-8b.md](recipes/maxtext-llama3.1-8b.md) runs
MaxText's Llama 3.1 8B with TE attention for 1,000 steps and lists what to
expect: step time, loss curve, startup and checkpoint costs, energy and ETTR.

[pipelines.md](pipelines.md#results-2026-11-05-8x-mi355x) has the step time,
memory and bubble of every schedule and layout on 8x MI355X. For Llama 3.1
8B with 65,536 tokens per step, the 2-stage zero-bubble pipeline takes 1.43 s
per step with 17 and 15 layers per stage, 1F1B takes 1.68 s, and the same
step without a pipeline takes 0.83 s with data parallelism and 0.80 s with
FSDP.
