# Pipelines

A JAX-Titan pipeline step is one Shardy MPMD program. Each stage is a named
computation on its own mesh, and each microbatch is an `mpmd.call` with its
own call counter. Shardy splits the step into fragments, one XLA program per
stage, microbatch and direction, and orders them with a pipeline schedule.
The whole step runs as one IFRT IR program; every stage keeps its own
optimizer state on its mesh.

## Schedules

`--schedule` names an entry of Shardy's pipeline registry. Every schedule
below matches single-mesh `jax.jit` in the CPU parity tests (`tests/`).

| Schedule | Order | Notes |
|---|---|---|
| `ZERO_BUBBLE_H1` (default) | 1F1B, with each backward split into an input-gradient and a weight-gradient fragment; weight gradients fill the bubble | Fastest on MI355X in every layout tried |
| `ZERO_BUBBLE_H2_*` | Zero-bubble with more forward fragments in flight | Same speed as H1 on 2 stages |
| `ONE_FWD_ONE_BWD` | 1F1B | |
| `GPIPE` | All forward fragments, then all backward fragments | Same speed as 1F1B on 2 stages |
| `GPIPE_BUT_1F1B_FOR_LAST_MESH` | GPipe, 1F1B on the last mesh | |
| `CIRCULAR`, `CIRCULAR_WITH_REVERSED_BACKWARD` | Several stages per mesh, all forward then all backward | Needs `--stages_per_mesh` > 1 |

With a schedule, JAX-Titan sets the Shardy option
`mpmd_merge_inferred_after_scheduling`. Without it, Shardy merges the
optimizer update into microbatch 0's backward fragment before scheduling,
so that fragment runs after every other backward fragment on every mesh.
jaxlib only passes this option on with
`third_party/patches/shardy-mpmd-parse-partitioning-options.patch`.

Shardy logs each mesh's fragment order when it compiles a step, in lines
starting `Module jit_train_step on mesh`. With 2 stages and 4 microbatches,
1F1B now runs F0 F1 B0 F2 B1 F3 B2 B3 on the first mesh and F0 B0 F1 B1 F2
B2 F3 B3 on the second.

## Inside a stage

Each stage mesh has a `data` and a `tensor` axis.

- **Data parallel** (default): each stage's parameters and optimizer state
  are replicated on its devices. `--reduce_grads=step` (default) gives each
  device its own parameter copy, so gradients are summed across devices once
  per step; `--reduce_grads=microbatch` all-reduces them in every
  microbatch's backward pass.
- **`--stage_fsdp`**: parameters and optimizer state are sharded across the
  stage's data axis. With `--reduce_grads=step`, each device all-gathers its
  parameter copy once per step and gradients are reduce-scattered once per
  step, as with ZeRO-2. With `--reduce_grads=microbatch`, weights are
  all-gathered and gradients reduce-scattered in every microbatch, as with
  FSDP (ZeRO-3).
- **`--tensor_parallel=N`**: Megatron-style tensor parallelism over the
  stage's `N` consecutive devices. Attention heads, MLP columns and the
  output vocabulary are sharded; `jaxtitan.parallel` derives the shardings
  from the logical axes `models/llama.py` gives each parameter.
- **`--stage_layers=17,15`**: decoder layers per stage. The last stage also
  runs the output head and loss, so giving it fewer layers balances the
  stages.

## Checkpoints

`--checkpoint_dir=DIR` saves parameters and optimizer state with Orbax after
the last step, and every `--checkpoint_every` steps; rerunning the same
command resumes from the latest checkpoint in `DIR`, including the data
position. Each stage's state is saved from, and restored onto, its own mesh.
Restoring places arrays on the shardings of the current command, so a run
can resume with another layout of the same stages, for example with
`--stage_fsdp` added.

- On CPU, a run resumed from step 3 matches the uninterrupted run exactly.
- On 8x MI355X, Llama 3.1 8B resumed from step 10 matches the uninterrupted
  run to 1e-7 at step 10 and exactly at step 11.
- A Llama 3.1 8B checkpoint is 80 GB. Saving it blocked training for 31-39 s
  while the arrays were copied to host memory, and the step after one save
  took 41 s while Orbax wrote it out in the background. Restoring from
  `/dev/shm` took 60 s. Checkpoint less often than every few minutes until
  saving is faster.

## Results (2026-11-05, 8x MI355X)

Llama 3.1 8B with 65,536 tokens per step (8 microbatches of 4 sequences of
2,048 tokens), full remat, XLA attention and AdamW on WikiText-103. Every run
starts from the same weights and trains on the same batches; all step-0
losses are 12.59. Step times are medians over steps 5-29. Per-step metrics
are in [results/2026-11-05](results/2026-11-05). The stack is jax
`6b39d495e5` and XLA `d23c30982a` with the patches in
`third_party/patches`.

Pipelines with 2 stages of 4 GPUs:

| Schedule | Inside each stage | Extra flags | Step | Tokens/s | MFU | Peak HBM per GPU |
|---|---|---|---|---|---|---|
| Zero-bubble H1 | DP, 16 and 16 layers | | 1.454 s | 45.1k | 10.5% | 180-181 GiB |
| Zero-bubble H1 | DP, 17 and 15 layers | `--stage_layers=17,15` | 1.426 s | 46.0k | 10.7% | 171-190 GiB |
| Zero-bubble H1 | FSDP, reduce per step | `--stage_fsdp` | 1.528 s | 42.9k | 10.0% | 141-143 GiB |
| Zero-bubble H1 | TP4 | `--tensor_parallel=4` | 1.788 s | 36.7k | 8.5% | 45-60 GiB |
| 1F1B | DP | `--schedule=ONE_FWD_ONE_BWD` | 1.684 s | 38.9k | 9.1% | 181-182 GiB |
| 1F1B | FSDP, reduce per microbatch | `--schedule=ONE_FWD_ONE_BWD --stage_fsdp --reduce_grads=microbatch` | 2.550 s | 25.7k | 6.0% | 43-47 GiB |
| GPipe | DP | `--schedule=GPIPE` | 1.692 s | 38.7k | 9.0% | 181-182 GiB |

Other pipeline shapes, and the same step without a pipeline:

| Layout | Flags | Step | Tokens/s | MFU | Peak HBM per GPU |
|---|---|---|---|---|---|
| Zero-bubble H1, 4 stages of 2 GPUs | `--num_stages=4` | 1.610 s | 40.7k | 9.5% | 81-105 GiB |
| Circular, 4 stages on 2 meshes of 4 GPUs | `--num_stages=4 --stages_per_mesh=2 --schedule=CIRCULAR` | 1.709 s | 38.3k | 8.9% | 164-171 GiB |
| No pipeline, data parallel over 8 GPUs | `--parallel=dp` | 0.833 s | 78.7k | 18.4% | 127 GiB |
| No pipeline, FSDP over 8 GPUs | `--parallel=fsdp` | 0.795 s | 82.4k | 19.2% | 26 GiB |

Compiling a 2-stage pipeline step takes 66-116 s; MPMD programs do not use
JAX's persistent compilation cache yet. The no-pipeline programs compile in
37-129 s and come from the cache after the first run.

### What the numbers show

- **Zero-bubble H1 is the fastest schedule**, 1.16x faster than 1F1B at the
  same memory. Giving the last stage one layer fewer adds 2%: that stage also
  runs the 128k-vocabulary output head and the loss, about 11% more FLOPs.
- **The bubble, from step times.** With 4 and 8 microbatches, step time is
  a fixed cost plus 0.16 s per microbatch for both schedules. The fixed cost
  is 0.47 s for 1F1B and 0.24 s for zero-bubble H1, 27% and 16% of the step
  at 8 microbatches. 1F1B's theoretical bubble with 2 stages and 8
  microbatches is 1/9 of the step; the rest of the fixed cost is
  once-per-step work, such as the gradient all-reduce and the optimizer
  update. 16 microbatches ran out of memory on the last stage. These runs
  predate the switch back to casting before the scan, which made steps
  2-3% faster.
- **The bubble, from traces.** In a zero-bubble trace, the last stage's GPUs
  are idle 14% of the time and the first stage's 32%; the first stage waits
  on the heavier last stage. Collectives take 10% and cross-mesh copies 1%.
- **FSDP inside stages** saves 40 GiB per GPU for 5% more step time when
  gradients are reduced once per step. Reducing them in every microbatch, as
  FSDP usually does, cuts memory to 43-47 GiB, a quarter of the data-parallel
  layout, at 1.5x the step time. Larger models need one of these to fit.
- **TP4 inside stages** uses a quarter to a third of the memory at 1.23x the
  step time. Tensor-parallel all-reduces are 22% of a TP2 trace, not
  overlapped with compute.
- **Cross-mesh transfers are cheap here.** Each microbatch moves 16 MiB of
  activations per GPU in about 0.29 ms (58 GB/s, one xGMI link), 0.3% of a
  step. rocprofiler reports them as `MemcpyH2D` on the receiving GPU, with
  no copy on the sender.
- **The pipeline is still 1.8x slower than FSDP on one node.** Besides the
  fixed per-step cost, each GPU runs 14.2k kernels per step against 4.0k for
  FSDP, because it processes one sequence at a time; the same FLOPs take more,
  smaller kernels. Pipelines are for models and clusters that outgrow FSDP,
  such as multi-node runs.
- **Tried and dropped:** casting the weights to bf16 inside the
  rematerialized layer, because the backward pass then casts again: 2-3%
  slower with 2 stages, 19% slower with 4, and 5-10 GiB more per GPU. XLA
  command buffers were 2% slower.

Earlier measurements on 2026-11-04 with 1F1B: reducing gradients once per
step instead of in every microbatch made the step 1.3x faster (2.20 s to
1.70 s) for 10-15 GiB more memory per GPU. Their logs are in
[results](results).

## MaxText and MaxDiffusion models (2026-11-06, 8x MI355X)

`--model=maxtext:llama3.1-8b` builds the stages from MaxText 0.2.5's own
layers, and `--model=maxdiffusion:flux-dev` from MaxDiffusion's FLUX.1
transformer (`jaxtitan.trainer.adapters`). Attention is their native JAX
implementation. Both reduce gradients in every microbatch's backward pass:
MaxText's layers constrain activation shardings, and FLUX.1's stages pass
tuples of activations, so neither has the per-rank parameter copies of
`--reduce_grads=step` yet. Step times are medians over steps 5-29 (5-19 for
FLUX.1), on the same stack as above. Per-step metrics are in
[results/2026-11-06](results/2026-11-06).

Llama 3.1 8B with 65,536 tokens per step on WikiText-103, 2 stages of 4 GPUs:

| Model code | Schedule | Gradients reduced | Step | Tokens/s | MFU | Peak HBM per GPU |
|---|---|---|---|---|---|---|
| MaxText | Zero-bubble H1 | per microbatch | 2.028 s | 32.3k | 7.5% | 177-178 GiB |
| MaxText | 1F1B | per microbatch | 2.013 s | 32.6k | 7.6% | 177-180 GiB |
| JAX-Titan's Llama | Zero-bubble H1 | per microbatch | 2.106 s | 31.1k | 7.3% | 167 GiB |
| JAX-Titan's Llama | Zero-bubble H1 | per step (default) | 1.459 s | 44.9k | 10.5% | 180-181 GiB |

FLUX.1-dev (11.9B parameters) with 32 images of 1024x1024 per step (8
microbatches of 4), each 4,096 latent and 512 text tokens, on synthetic
rectified-flow batches, 4 stages of 2 GPUs with 15, 14, 14 and 14 blocks and
zero-bubble H1: 7.269 s per step, 4.4 images/s (20.3k tokens/s) and 4.9% MFU,
with 218, 120, 88 and 89 GiB peak HBM per GPU from the first stage to the
last. The loss falls from 3.51 to 2.75 in 20 steps.

- **MaxText's layers are as fast as JAX-Titan's Llama** when both reduce
  gradients per microbatch: 4% faster here, for 10 GiB more memory per GPU.
  MaxText initializes differently (step-0 loss 12.30 against 12.58); both
  reach 7.2-7.5 after 30 steps.
- **The gap to JAX-Titan's default, 1.39x, is the gradient reduction:**
  all-reducing each stage's gradients in all 8 microbatches instead of once
  per step. With that reduction, 1F1B and zero-bubble are within 1%.
- **FLUX.1-dev does not fit in 2 stages of 4 GPUs.** The first of 2 stages
  holds all 19 double-stream blocks, which carry 2.4 times the weights of a
  single-stream block: 7.9B of the 11.9B parameters. With fp32 weights,
  AdamW state and per-microbatch gradients, its GPUs reached 272 GiB at step
  0 and ran out of memory at step 1, even with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. With 4 stages the first holds 5.1B.
- Compiling took 53-54 s for the MaxText pipelines, 74-110 s for JAX-Titan's
  Llama and 123 s for FLUX.1.

## Operator providers (2026-11-06, 8x MI355X)

`--ops` routes semantic ops to ROCm TE (the 2026-11-02 nightly wheel) or
JAX-AITER (the `third_party/jax-aiter` submodule, built by
`scripts/build_jax_aiter.sh`); everything else stays native JAX. Before any
training run, `python -m jaxtitan.ops.conformance` checked every op against
native JAX in bf16 and float32 on Llama 3.1 8B and FLUX.1 shapes; its rows
are in [results/2026-11-06](results/2026-11-06) (`ops-conformance_*.jsonl`).

| Op | Provider | One GPU | vmap | Sharded | shard_map | MPMD step | Forward plus backward |
|---|---|---|---|---|---|---|---|
| attention | `te` | pass | pass | pass | pass | pass | 3.7x (Llama), 3.3x (FLUX.1) |
| attention | `aiter` | pass | no batching rule | pass | pass | pass | 2.7x (Llama), 2.2x (FLUX.1) |
| rms_norm | `aiter` | pass | wrong forward | pass | pass | pass | 0.96x |
| rms_norm | TE (not offered) | pass | wrong forward and gradient | wrong weight gradient | pass | wrong weight gradient | 0.72x |
| swiglu | `aiter` (SiLU-and-mul) | pass | pass | pass | pass | pass | 0.94x |
| linear | `te`, `aiter` (BF16) | pass | no batching rule | pass | pass | pass | 0.93-1.05x |
| linear, swiglu | `te_fp8`, `te_mxfp8` | pass (FP8 error) | broken batching rule | pass | pass | pass | 0.6-1.3x |
| linear, swiglu | `aiter_mxfp4` | wrong (uncorrelated output) | | wrong | wrong | wrong | |

Providers decline what failed (vmap cases, and JAX-AITER attention in a
process with TE), so those calls fall back to native. The FP8 and MXFP8
GEMMs are within 3.7-3.8% of BF16 per GEMM and 6.7% for SwiGLU's three.

Training with the same layouts and batches as above:

| Model | Gradients reduced | `--ops` | Step | Throughput | MFU |
|---|---|---|---|---|---|
| Llama 3.1 8B, 2 stages | per step (vmap) | native | 1.457 s | 45.0k tokens/s | 10.5% |
| Llama 3.1 8B, 2 stages | per step (vmap) | `attention=te` | 1.260 s | 52.0k tokens/s | 12.1% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | native | 1.688 s | 38.8k tokens/s | 9.1% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `attention=te` | 1.491 s | 44.0k tokens/s | 10.3% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `attention=aiter` | 1.529 s | 42.9k tokens/s | 10.0% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `attention=te` + `te_fp8` GEMMs | 1.676 s | 39.1k tokens/s | 9.1% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `attention=te` + `te_mxfp8` GEMMs | 1.747 s | 37.5k tokens/s | 8.7% |
| FLUX.1-dev, 4 stages | per microbatch | MaxDiffusion's attention | 7.269 s | 4.4 images/s | 4.9% |
| FLUX.1-dev, 4 stages | per microbatch | `attention=native` (XLA) | 8.001 s | 4.0 images/s | 4.5% |
| FLUX.1-dev, 4 stages | per microbatch | `attention=te` | 5.764 s | 5.6 images/s | 6.2% |
| FLUX.1-dev, 4 stages | per microbatch | `attention=aiter` | 6.028 s | 5.3 images/s | 5.9% |

- **TE fused attention is the biggest win:** 1.16x on Llama 3.1 8B (12.1%
  MFU) and 1.26x on FLUX.1-dev, at the same memory. JAX-AITER's flash
  attention gets 1.10x and 1.21x. Loss curves track native.
- **`--per_device=shard_map` costs 16%:** Shardy MPMD does not split a
  backward fragment that contains a manual computation into its activation
  and weight gradients, so zero-bubble runs like 1F1B (16 split stage-1
  backward fragments with vmap, 8 with shard_map). TE attention works under
  vmap and should use it; JAX-AITER needs shard_map until its kernels have
  batching rules or Shardy splits manual computations.
- **FP8 and MXFP8 GEMMs are slower end to end** (1.68 and 1.75 s against
  1.49 s in BF16): quantizing every GEMM's inputs in forward and backward
  costs more than the faster GEMMs save at Llama 3.1 8B's sizes.
- **Not usable yet:** JAX-AITER's MXFP4 GEMM returns output uncorrelated with
  the BF16 product (relative error 1.4) on this stack, though its own tests,
  which check finiteness, pass. JAX-AITER exposes no MoE kernels at its pin,
  and `flash_attn_varlen` waits for a packed-sequence data path.
