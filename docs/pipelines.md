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
