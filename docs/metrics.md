# Training metrics

JAX-Titan records the training metrics of the ROCm scaling book (the
rocm-dev-infra build plan): the definitions in its A0 methodology, and what
its Chapter 9 (multi-node training), Chapter 6 (communication) and Chapter 13
(debugging) measure on a training run. A run directory holds the raw records;
`jax_titan.observability` turns them into each metric.

## Recording a run

```bash
export JAX_TITAN_LAUNCH_TIME=$(date +%s.%N)   # optional: counts launch time in u0
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=llama3.1-8b --num_stages=2 --steps=100 \
  --run_dir=runs/llama8b_pp2 \
  --eval_every=50 \
  --checkpoint_dir=runs/llama8b_pp2/ckpt --checkpoint_every=50 \
  --profile_steps=20:22 --profile_dir=runs/llama8b_pp2/trace
```

`--run_dir` follows the book's `runs/<cell>/` layout:

| File | Contents | Written by |
|---|---|---|
| `train.jsonl` | One record per step (loss, gradient norm, step time, tokens/s, tokens/s per GPU, MFU, wall time) and timeline events: `start`, `layout`, `state`, `compile`, `data`, `memory`, `eval`, `save`, `checkpoint_closed` and `end` (`jax_titan.observability.records`). A restart appends a new attempt. Processes past the first write `train.p<i>.jsonl` | the trainer |
| `env.json` | Packages, git commits, ROCm; XLA flags and the RCCL, HSA, HIP and JAX environment; host, CPUs, GPUs (PCI bus ID, VBIOS, driver, power cap), RDMA NICs; the rank map (device, process, HIP index, PCI bus ID, mesh and stages) | the trainer, or `python -m jax_titan.observability.env OUT.json` |
| `power.jsonl` | Per GPU every `--power_sample_s` seconds (default 1): energy counter, power, GFX and memory clocks, hotspot temperature; every 10 s the share of time throttled by the power limit, temperatures or PROCHOT | the trainer, or `python -m jax_titan.observability.power OUT.jsonl` beside any job |
| `profile.json` | Per GPU: compute, exposed collective, copy and idle shares of the traced steps, exposed collective seconds per step, and the spread across GPUs | the trainer with `--profile_steps`, or `python -m jax_titan.observability.profile TRACE --steps N --json OUT` |
| `rccl_inspector/` | Every collective and send/receive with its completion time, from the RCCL Inspector | the trainer with `--rccl_inspector` |
| `results.json` | Every metric below for this run | the trainer at the end, or `summarize` |

`--rccl_inspector` needs the Inspector plugin, which RCCL carries as CUDA-only
source; `scripts/build_rccl_inspector.sh` ports and builds it for HIP into
`$JAX_TITAN_ROOT/lib`. `--eval_every` validates on `--eval_batches` fixed
batches of WikiText-103's validation split (held-out batches for synthetic
data), the same batches every pass.

## Tools

```bash
# Per run: results.json, and across runs: repeats, scaling, Pareto front, pipeline fit.
python -m jax_titan.observability.summarize runs/a runs/b ... \
  [--skip 5] [--ceilings ceilings.json] [--target_loss 3.0] [--json report.json]

# Gate 0: each run's losses over the first K steps against a reference run.
python -m jax_titan.observability.summarize --gate0 runs/reference runs/a runs/b \
  --gate0_steps 10 --gate0_rtol 2e-3

# ETTR, effective throughput, and the checkpoint interval and ETTR projected to 2,048 nodes.
python -m jax_titan.observability.ettr runs/a --json runs/a/ettr_model.json

# Calibrated ceilings of one GPU (BF16 and FP8 GEMM, HBM) for calibrated MFU.
HIP_VISIBLE_DEVICES=0 python -m jax_titan.observability.ceilings ceilings.json

# Step-time overhead of each failure detector.
python -m jax_titan.observability.detectors runs/detectors -- --model=llama3.1-8b --steps=30

# Collective completion times from an Inspector directory.
python -m jax_titan.observability.collectives runs/a/rccl_inspector

# The GEMMs a run's compiled program launches: shapes, dtypes, layouts, library and
# kernel configuration, launches per training step (below). Standard library only.
python3 src/jax_titan/observability/gemms.py runs/a.hlo --run_dir runs/a --merge [--sites] [--csv gemms.csv]

# Figures: MFU and step time over wall time with saves, validation and restarts marked,
# loss against tokens, gradient norm, power; comparison.svg for several runs. Needs the
# plots extra (matplotlib); the trainer draws <run_dir>/figures/ itself when it is there.
python -m jax_titan.observability.plot runs/a runs/b --out figures

# Training FLOPs and communication per op and per stage, and the balanced layer split.
python -m jax_titan.observability.flops --model=llama3.1-8b --num_stages=2 \
  --peak_tflops=1436 --busbw_gbs=160

# Before a run: its time, GPU-hours, energy, memory and data-parallel communication on
# other GPU types and counts, from a run's measurements (docs/planning.md).
python -m jax_titan.planning.budget --model=llama3.1-8b --tokens=1e12 \
  --gpu=MI325X,MI355X --gpus=1024 --seq_len=2048 --from_run=runs/a

# A run's step part by part against its prediction, a fit, and the fit at other GPU
# counts; records go to a performance database that flags regressions.
python -m jax_titan.planning.step --calibrate runs/a --ladder 8,64,256,512 \
  --ladder_batch_tokens 4194304 --perfdb perf.jsonl
python -m jax_titan.observability.perfdb perf.jsonl

# Collective bus bandwidth by message size on this node (RCCL; MORI-EP when installed),
# and a running job's time left.
bash scripts/bench_collectives.sh
python -m jax_titan.planning.remaining runs/a --target_tokens 2e12
```

The operator conformance bench (`jax_titan.ops.conformance --modes=bench`) ends
with each provider's geometric-mean speedup across shapes and its
time-weighted speedup (total native time over total provider time).

## GEMMs in the compiled program

`jax_titan.observability.gemms` lists every GEMM the compiled program
launches, with the library and kernel configuration XLA chose for it. It reads
XLA's dump of each module's final HLO,
`module_*.<gfx arch>_gpu_after_optimizations.txt`, written after every pass and
after autotuning has bound each GEMM to the kernel it picked. The kernels the
autotuner timed and dropped never appear: `autotune_results.pbtxt` and the
`before_*` dumps are not read. When XLA also wrote the module's
`thunk_sequence.txt`, the executable's launch list, the tool checks that it
launches every GEMM the HLO holds.

The tool is one file that uses only the Python standard library. It needs
neither JAX nor JAX-Titan, reads the dump of any XLA program, and runs from a
checkout or as a copy on another machine with Python 3.8 or later:
`python3 src/jax_titan/observability/gemms.py DUMP`. Inside JAX-Titan's
environment, `python -m jax_titan.observability.gemms` runs the same code.

The parallelism does not matter either. Under SPMD (`--parallel=dp` or
`fsdp`, or any `jax.jit` program) XLA compiles one partitioned module,
`jit_train_step`, which a step runs once on every device. Under MPMD it
compiles one module per pipeline fragment, which runs on its stage's devices.
Either way the shapes are what one device computes, and the merged table of a
pipeline adds up one device of each stage.

Dump while the program compiles; with a warm compilation cache XLA compiles
nothing and dumps nothing. Two steps suffice, and the dump of the 8B pipeline
is 18 MB:

```bash
JAX_ENABLE_COMPILATION_CACHE=false \
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS --xla_dump_to=runs/a.hlo --xla_dump_hlo_as_text" \
  python -m jax_titan.trainer.train --model=llama3.1-8b --num_stages=2 --steps=2 \
  --run_dir=runs/a
python3 src/jax_titan/observability/gemms.py runs/a.hlo --run_dir runs/a --merge
```

`runs/a.hlo` is the dump, one set of files per compiled module. `--run_dir
runs/a` points at the trainer's run directory, whose `train.jsonl` says how
often one training step runs each module: the counts become launches per step,
and the modules a step never runs (initialization, random-number helpers) drop
out. `--merge` prints one table over all modules, adding up a shape that
several pipeline fragments run; without it there is one table per module.

The dump's HLO is the same on every ROCm GPU, so the tool reads MI300X and
MI325X dumps (`gfx942_gpu_after_optimizations.txt`, with FP8 in the `fnuz`
types) as it reads MI355X's; what changes is the library and configuration
the autotuner picks for each GEMM. The results below are from gfx950.

A GEMM is a hipBLASLt call (`__cublas$lt$matmul`, also `$f8` and `$mx`), a
rocBLAS call (`__cublas$gemm`), a fusion around a `dot` (`Triton` for XLA's
Triton GEMMs), or a plain `dot`, `ragged-dot` or `scaled-dot` (`XLA`). One in
a `while` body counts once per trip when XLA knows the trip count, as for the
scan over each stage's layers; one in a branch of a `conditional` is flagged.
Without `--run_dir`, counts are per run of each module. With it, they are per
training step: the trainer's `compile` event records each program a step runs
and its calls (`programs` in `train.jsonl`). For a program the trainer did not
run, `--calls FILE` takes the same counts from a JSON object
`{"module": calls}`, or from the step's `lowered.as_text()`, whose IFRT or MPMD
fragment calls it counts. An SPMD step runs `jit_train_step` once. A pipeline
runs each stage's forward fragment once per microbatch (`p1_stage1_fwd_call0`, ...), and a
backward fragment serving all microbatches (`p3_stage1_bwd_calls0to7`) once
for each; the name alone does not give the count, since `p7_stage1_bwd` also
runs once per microbatch.

The outputs below come from Llama 3.1 8B on two MI355X GPUs, one per stage,
with one sequence per microbatch. Each GPU gets the GEMM shapes of the default
2 stages x 4 GPUs, where `--microbatch_size=4` splits a microbatch's four
sequences across a stage's four GPUs:

```bash
HIP_VISIBLE_DEVICES=0,1 JAX_ENABLE_COMPILATION_CACHE=false \
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS --xla_dump_to=runs/llama8b_pp2.hlo --xla_dump_hlo_as_text" \
  python -m jax_titan.trainer.train --model=llama3.1-8b --num_stages=2 --microbatches=8 \
  --microbatch_size=1 --seq_len=2048 --steps=3 --run_dir=runs/llama8b_pp2
python3 src/jax_titan/observability/gemms.py runs/llama8b_pp2.hlo --run_dir runs/llama8b_pp2 --merge
```

The top rows of one training step:

```
19 modules, in one training step: 7448 GEMM launches, 975.2 TFLOP; FLOPs by library: hipBLASLt 68% (4120 launches), Triton 32% (3328 launches)
  thunk sequence: 96 of 96 fused GEMM kernels by name, 70 library GEMM thunks for 70 library calls (all launched)
   count  library    dtype             batch       M       N       K  layout    GFLOP  share  pass      source
     512  Triton     bf16                  -    2048   14336    4096  MK,KN       241  12.6%  fwd       native.py:22 swiglu
     256  Triton     bf16                  -    2048    4096   14336  MK,KN       241   6.3%  fwd       native.py:22 swiglu
     256  hipBLASLt  bf16                  -    2048   14336    4096  MK,KN       241   6.3%  remat     native.py:22 swiglu
     256  hipBLASLt  bf16                  -    2048    4096   14336  MK,NK       241   6.3%  bwd       native.py:22 swiglu
     256  hipBLASLt  bf16                  -    4096   14336    2048  KM,KN       241   6.3%  bwd       native.py:22 swiglu
     ...
     384  hipBLASLt  bf16                  -    2048    6144    4096  MK,KN       103   4.1%  fwd,remat native.py:27 linear
     512  hipBLASLt  bf16->f32             8    8192    2048     128  MK,NK      34.4   1.8%  fwd,remat native.py:16 attention
       8  hipBLASLt  bf16->f32             -    2048  128256    4096  MK,KN  2.15e+03   1.8%  fwd       llama.py:257 stage_forward
```

Each row is one kernel shape, by total FLOPs:

| Column | Meaning |
|---|---|
| `count` | Launches per run of the module, or per step with `--run_dir`: the 512 of the top row are 8 microbatches x 32 layers x 2, for the gate and up projections |
| `library` | hipBLASLt, rocBLAS, Triton, or XLA for a plain `dot` |
| `dtype` | Operand types and, when it differs, the output's: `bf16->f32` |
| `batch` | The GEMM's own batch dimension, for a batched GEMM such as attention's 8 KV-head groups; `-` for a plain GEMM, `gN` for a ragged dot over N groups. The training batch is not here: a projection's M is the tokens one GPU processes in a microbatch (1 sequence x 2048 above), and each microbatch is another launch in `count` |
| `M`, `N`, `K` | As the kernel computes C[M, N] row major; when XLA writes C column major, the operands swap roles |
| `layout` | Which dimension of each operand is contiguous: `MK` is A[M, K] with K contiguous, `KM` is A stored as [K, M]; `KN` and `NK` likewise for B. `MK,NK` is X times W^T, the backward GEMM for the input gradient |
| `GFLOP` | 2 x batch x M x N x K for one launch |
| `pass` | From JAX's op name: `fwd`, `bwd`, or `remat` for a forward op recomputed in the backward pass |
| `source` | The innermost Python frame of the op |

The library comes from the instruction XLA emitted: hipBLASLt for a
`__cublas$lt$matmul` custom call, which the executable runs as a
`kCublasLtMatmul` thunk, and Triton for a `__triton_nested_gemm_fusion` (or the
older `__triton_gemm`) fusion, which it launches as a kernel of the same name.
`--sites` lists each GEMM instruction with its configuration, here for the
stage 1 forward fragment of the first microbatch in the run above:

```bash
python3 src/jax_titan/observability/gemms.py runs/llama8b_pp2.hlo --run_dir runs/llama8b_pp2 \
  --module p1_stage1_fwd_call0 --sites
```

```
p1_stage1_fwd_call0.train_step (module_899739.p1_stage1_fwd_call0.train_step.gfx950_gpu_after_optimizations.txt), runs 1 time per step
  8 GEMMs, 113 launches, 17.54 TFLOP per step; FLOPs by library: Triton 75% (80 launches), hipBLASLt 25% (33 launches)
  thunk sequence: 5 of 5 fused GEMM kernels by name, 3 library GEMM thunks for 3 library calls (all launched)
   count  kernel                     library          M       N       K  layout pass   config                                                 source
      16  gemm_fusion_dot.7          Triton        2048   14336    4096  MK,KN  fwd    128x128x64, warps 4, stages 2, grid 1792, LDS 66 KiB   native.py:22 swiglu
      16  gemm_fusion_dot.6          Triton        2048   14336    4096  MK,KN  fwd    128x128x64, warps 4, stages 2, grid 1792, LDS 66 KiB   native.py:22 swiglu
      16  gemm_fusion_dot.8          Triton        2048    4096   14336  MK,KN  fwd    128x256x64, warps 8, stages 2, grid 256, LDS 100 KiB   native.py:22 swiglu
       1  custom-call.1              hipBLASLt     2048  128256    4096  MK,KN  fwd    algorithm 3, workspace 64 MiB                          llama.py:257 stage_forward
      16  custom-call.0              hipBLASLt     2048    6144    4096  MK,KN  fwd    algorithm 0, workspace 64 MiB                          native.py:27 linear
      16  gemm_fusion_dot.5          Triton        2048    4096    4096  MK,KN  fwd    256x128x64, warps 8, stages 2, grid 256, LDS 99 KiB    native.py:27 linear
      16  custom-call.2              hipBLASLt     8192    2048     128  MK,NK  fwd    algorithm 73, alpha 0.08839, workspace 64 MiB          native.py:16 attention
      16  gemm_fusion_dot.12         Triton         128    8192    2048  KM,NK  fwd    128x256x64, warps 8, stages 3, grid 256, LDS 148 KiB   native.py:16 attention
```

| Library | Configuration from the HLO |
|---|---|
| Triton | `block_m x block_n x block_k` of C[M, N] row major, `num_warps` and `num_stages`, and any other tiling setting off its default (`mfma_size`, `waves_per_eu`, `split_k` for `__triton_gemm`). From `thunk_sequence.txt`: the grid, ceil(M / block_m) x ceil(N / block_n) x batch workgroups, and the LDS bytes |
| hipBLASLt, rocBLAS | `algorithm`, the index of the chosen candidate among those hipBLASLt's heuristic returns for the problem, which the autotuner timed (0 is hipBLASLt's first pick); the workspace; and when set, the epilogue (bias, activation), `alpha` (the attention's 1/sqrt(128) above), `beta`, and `scale_mode` for FP8 and MX |

The HLO does not name the hipBLASLt kernel the algorithm index selects; the
kernel names appear in a profiler trace of the run. `--module
REGEX` keeps some modules; `--csv` and `--json` write every GEMM with its
operand shapes, call stack, configuration and launch dimensions. Custom calls
it does not read as GEMMs, such as the operator providers' kernels
(`--ops=linear=aiter`), are listed with their launches.

The step's GEMMs agree with the trainer's FLOPs: 263.5 TFLOP forward, 527.0
backward and 184.7 recomputed. Forward and backward come to 790.5 TFLOP, 26.3
above the trainer's model FLOPs (764.2), which count causal attention at half:
the native attention computes all of QK^T and PV, 52.8 TFLOP of GEMMs, half of
which is 26.4. The trainer's hardware FLOPs (1001.6) assume the whole forward
is recomputed; XLA recomputed 70% of it.

## The metrics

| Scaling-book metric | Where JAX-Titan records it | Computed by |
|---|---|---|
| Tokens/s and tokens/s per GPU | step records | `summarize` |
| MFU, against the nominal peak and the calibrated ceiling | step records (nominal, `--peak_tflops_per_device`); `ceilings.json` | `summarize --ceilings` |
| MFU sustained, training-active, end to end and rolling; HFU | step records, `layout` event (model and hardware FLOPs per step) | `summarize`, `ettr`, `plot` |
| Aggregate MFU of several runs | runs | `summarize` |
| Training FLOPs per op and stage (model and executed), communication volume per GPU, balanced stage split | model config and layout | `flops` |
| Kernel speedups: geometric mean across shapes, time-weighted | bench rows | `jax_titan.ops.conformance` |
| MFU, loss and power over time, with events | step records and events, `power.jsonl` | `plot` |
| Scaling efficiency `E(N) = T(N) / ((N/N0) T(N0))` | runs at several device counts | `summarize` |
| Step time p50, p95 and max across steps | step records | `summarize` |
| Step time across ranks; straggler spread | one log per process; per-GPU shares and clocks | `summarize` (process spread), `profile` (spread across GPUs), `power` (clocks) |
| Exposed communication | trace | `profile`: collective kernel time no compute kernel overlaps, per GPU and step |
| Collective completion time p50 and p99 | RCCL Inspector | `collectives`, `summarize` |
| Checkpoint cost `w_cp` and restore time | `save` and `state` events, and the steps after each save | `summarize`, `ettr` |
| Restart overhead `u0`, by stage | `start`, `state`, `compile`, `data` events and the first step | `summarize` (`startup_s`), `ettr` |
| ETTR and its loss terms; effective training throughput | events across attempts in one log | `ettr` |
| Checkpoint interval (Daly) and ETTR at 1-2,048 nodes | measured `w_cp` and `u0`, published failure rates | `ettr --json` (the book's `ettr_model.json`) |
| Energy per token (GPU boundary), clocks, throttle reasons | `power.jsonl` | `summarize` |
| Pipeline utilization | runs at two or more microbatch counts; per-stage idle in traces | `summarize` (pipeline fit), `profile` |
| Time to a target validation loss | `eval` events | `summarize --target_loss` |
| Gate 0: every layout within a stated tolerance of a reference over the first K steps | step records of two runs | `summarize --gate0` |
| Median and spread of repeats; regression threshold from their variance | three or more runs of one configuration | `summarize` |
| Co-design Pareto front: tokens/s per GPU against peak HBM | `memory` events | `summarize` |
| Detector overhead | one run per detector | `detectors` (`detector_overhead.json`) |
| `env.json` and the rank map | at the start of every run | `env` |

## Definitions

- **Steady steps.** Each attempt's first `--skip` steps (default 5) and the
  `--wcp_steps` steps after each save (default 2) are left out of step-time
  statistics. Step time is the wall time of the compiled step, from dispatch
  until the loss reaches the host.
- **MFU** is model FLOPs per step (6N plus causal attention: a matmul's
  backward costs twice its forward, so training is three forward passes) over
  step time times devices times the peak: 2,500 TFLOP/s BF16 for MI355X
  nominally, or the BF16 GEMM rate `ceilings` measures. Over many steps it is
  always total model FLOPs over total elapsed time, never a mean of per-step
  MFUs, which would weigh a slow step like a fast one. `summarize` reports it
  at the median step; **sustained**, over the steady steps; **training-active**,
  over every step the trainer ran; **end to end**, over the committed steps and
  the wall time of all attempts (startup, saves, validation, lost work and
  relaunch included, so it is ETTR times the steady MFU); and **rolling**, over
  each window of 50 consecutive steps past an attempt's first step.
- **HFU** counts the FLOPs the GPUs execute: with rematerialized layers (the
  default `--remat`), every decoder layer's forward pass again, 1.31x the
  model FLOPs for Llama 3.1 8B at 2,048 tokens.
- **Speedups across kernel shapes.** The geometric mean weighs every shape the
  same; the time-weighted speedup weighs them by time. Neither is the gain of a
  training step, which depends on that kernel's share of the step: TE's
  operators give 1.56x as a geometric mean and 2.60x time-weighted over their
  bench shapes, and TE attention made the Llama 3.1 8B step 1.16x faster.
- **Exposed communication** is the time a GPU runs collective kernels while no
  compute kernel runs, from a `jax.profiler` trace. Summed kernel durations
  are not step time.
- **`w_cp`** is how long the step loop waits for a save, plus the slowdown of
  the next `--wcp_steps` steps against the steady step time (Orbax finishes
  writing in the background).
- **`u0`** runs from the launch (`JAX_TITAN_LAUNCH_TIME`, else process start)
  until the first step ends, minus one steady step. Its stages: launch, Python
  imports, JAX init, setup (model and layout), state init or restore, compile,
  data, the first step's excess over a steady step (RCCL communicator setup and
  first-run work), and other.
- **ETTR** (Meta FAIR, arXiv 2410.21680) is productive time over wall time.
  Productive time is the committed steps times the steady step time; committed
  steps are those before an attempt's last save, and all of the last attempt.
  The remainder is `u0`, `w_cp`, lost work (steps redone after a restart),
  validation, queue time and other. **Effective training throughput** is the
  steady tokens/s times ETTR, which is committed tokens over wall time.
- **Checkpoint interval.** Daly's optimum
  `sqrt(2 d M) [1 + sqrt(d / 2M) / 3 + d / 18M] - d` for checkpoint cost `d`
  and mean time to failure `M = 1000 / (rate x nodes)` days, with failure
  rates of 2.5-17.5 per 1,000 node-days (Meta FAIR) and about 3.8 (Llama 3).
  The projected ETTR is `tau / (tau + d) x (1 - (R + tau / 2) / M)` with
  restart cost `R`. The projection assumes the measured step time holds.
- **Energy per token** is the GPUs' energy counters over the steady window
  divided by the tokens trained in it. The boundary is the GPU: no CPU, NIC,
  memory or cooling. Sampling at 1 Hz cannot see millisecond power excursions.
- **Pipeline utilization** is `1 - fixed / step time`, where the fixed
  per-step cost comes from a least-squares fit of step time against the
  microbatch count over runs that differ only in microbatches.
- **Gate 0** compares the loss of each step over the first K steps with a
  reference run of the same weights and batches: the no-pipeline layout
  (`--parallel=dp`) for pipelines, or a one-node run for multi-node layouts.
  A run passes when every relative difference is within the tolerance.

## Measured on 8x MI355X

- **Ceilings** (one GPU, through XLA): BF16 GEMM 1,436 TFLOP/s at
  16384x16384x16384 (57% of the nominal 2,500), FP8 GEMM 2,455 TFLOP/s, HBM
  stream 5.74 TB/s. MFU against the calibrated BF16 ceiling is 1.74x MFU
  against the nominal peak.
- **Gate 0, MaxText Llama 3.1 8B** (2 stages x 4 GPUs, zero-bubble, against
  `--parallel=dp` on 8 GPUs, same weights and batches): over the first 10
  steps every pipeline stays within 8.7e-4 relative loss of the reference
  (MaxText's attention 6.3e-4, TE 8.7e-4, JAX-AITER 8.7e-4; step 0 within
  1.1e-5). After the warmup's loss spikes (steps 2-10, gradient norms up to
  54), bf16 rounding differences grow to 1-7%, so K = 10 with a tolerance of
  2e-3 is the gate on one node.
- **A recorded run** ([results/2026-11-07/llama3.1-8b_pp2_run-dir](results/2026-11-07/llama3.1-8b_pp2_run-dir)),
  JAX-Titan's Llama 3.1 8B on 2 stages of 4 GPUs: step time p50 1.703 s, p95
  1.761 s; MFU 9.0% nominal, 15.6% calibrated; startup 153 s (compile 113 s,
  first step's excess 27 s, init 9 s); 0.245 J/token at the GPUs (4.6 kW for
  8 GPUs, GFX clocks 2.0-2.1 GHz); about 140 ms of exposed collective time per
  GPU per step, with the first stage's GPUs idle 41% of the traced steps and
  the last stage's 24%.
- **RCCL Inspector** (ported to HIP, GPU kernel timing): each stage's
  gradient all-reduce over 4 GPUs takes 30-80 ms at p50 for 2.4-8.5 GB and up
  to 183 ms at p99, at 123-171 GB/s bus bandwidth
  ([collectives](results/2026-11-07/llama3.1-8b_pp2_rccl-inspector_collectives.json)).
  MPMD's cross-mesh transfers do not go through RCCL, so the Inspector does
  not see them.
- **Detector overhead** (`detectors --repeats 3`, MaxText Llama 3.1 8B on 2
  stages of 4 GPUs; the baseline's three runs lie within 0.2%,
  [table](results/2026-11-07/maxtext-llama3.1-8b_pp2_detector-overhead.json)):

  | Detector | Step p50 | Overhead |
  |---|---|---|
  | Baseline (XLA stuck and terminate timeouts and RCCL RAS on, as by default) | 2.098 s | |
  | Step watchdog | 2.097 s | 0.0% |
  | RCCL RAS (measured by turning it off) | 2.098 s | 0.0% |
  | RCCL Inspector | 2.095 s | -0.1% |
  | ROCr Debug Agent | 2.098 s | 0.0% |
  | ROCr Debug Agent with precise ALU exceptions | 2.112 s | +0.7% |
  | XLA progress tracking (16 thunks) | 2.285 s | +8.9% |
  | XLA NaN/Inf checks, checksum tracing | fails | |

  Everything but progress tracking can stay on; progress tracking is for
  reproducing a hang. XLA's NaN, Inf and checksum checks fail on any program
  that returns a tuple ("Module output slices must not contain tuple
  shapes"): `GetModuleOutputSlices` in `gpu_executable.cc` lists the result
  tuple itself, at upstream XLA's head too, so they need an XLA fix first.

- **MFU variants of the recipe run** ([recipe](recipes/maxtext-llama3.1-8b.md),
  1,000 steps with a rehearsed restart): 8.10% at the median step, 8.12%
  sustained, 7.99% training-active, 5.99% end to end (ETTR 0.739), 6.9-8.2%
  over 50-step windows.
- **FLOPs model, Llama 3.1 8B on 2 stages** (`flops`): 46.6 GFLOP per token,
  3.06 PFLOP per step of model FLOPs and 4.01 PFLOP executed with
  rematerialization. The last stage carries the output head, 11% more executed
  FLOPs than the first with 16 and 16 layers, so `--stage_layers=17,15`
  balances them, the split measured fastest on 2026-11-05 (1.426 s against
  1.454 s). At XLA's 1,436 TFLOP/s the stages' compute takes 331 and 367 ms of
  the 1.50 s step. Each GPU all-reduces 24 GB of gradients per step, about
  150 ms at the 160 GB/s the Inspector measured, close to the 140 ms of
  exposed collective time in traces; with gradients reduced in every
  microbatch, as MaxText's layers are today, it is 193 GB.

## Not covered yet

These need more than one node, admin rights, or other chapters' deliverables:

- NIC hardware counters at about 1 ms (the book's `nic_sampler.py`), for the
  capped-rank straggler signature, and fabric placement in `env.json` (M5).
- Step time across nodes and RCCL communicator setup by node count (M5, Q1);
  today the first step's excess holds communicator setup.
- Power caps below the default (P2), which need admin rights.
- Per-op kernel times for the step-time projection (the book's `kernel_db`,
  Chapter 5). `planning.collectives` is the communication model (Chapter
  6's `comm_model.json`), measured on one MI355X node. `planning.step` sets
  predicted parts against a run's, but its compute is one efficiency for
  every kernel.
- Time to detect and to localize injected faults, and their accuracy (DBG1).
