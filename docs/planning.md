# Planning a training run

`jax_titan.planning.budget` estimates a training run before it starts: its
FLOPs, time, GPU-hours, energy and cost, whether its layout fits in memory,
and whether its data-parallel communication hides behind compute. Give it a
model, a token or time budget, and one GPU type or several to compare.

## Examples

Each example below is the command and what it prints. They run on the CPU in
a second or two, with the defaults: 40% MFU, 8,192-token sequences, 4.19M
tokens per step, fp32 AdamW state and x2.00 memory overhead. After the first,
the outputs show only the lines that example is about.

### Tokens on one GPU type: the full report

```bash
python -m jax_titan.planning.budget --model=llama3.1-8b --tokens=2e12 --gpu=MI325X --gpus=256
```

```text
--------- Model Overall Report ---------
  model:                  llama3.1-8b
  config source:          unsloth/Meta-Llama-3.1-8B@e9a141a2091e
  parameters:             8.0B
  FLOPs per token:        51.5 GFLOP at sequence 8192

--------- System Budget Full Report ---------
  GPUs:                   256 x AMD Instinct MI325X
  peak:                   1307 TFLOP/s bf16 dense
  MFU:                    40.0% (assumed)
  HFU:                    52.5%
  ETTR:                   0.997
  checkpoint interval:    0.51 h (Daly)
  tokens:                 2.000T (249 per active parameter)
  model FLOPs:            1.029e+23
  executed FLOPs:         1.352e+23
  compute-optimal:        28.3B parameters on 0.57T tokens (Chinchilla, C = 6ND)
  throughput:             2.60M tokens/s
  throughput per GPU:     10160 tokens/s
  wall time:              8.9 days
  stepping time:          8.9 days
  GPU-hours:              54.83k
  energy:                 55 MWh at 100% of board power
  wall time at 10% MFU:   35.7 days
  wall time at 20% MFU:   17.8 days
  wall time at 30% MFU:   11.9 days
  wall time at 40% MFU:   8.9 days
  wall time at 50% MFU:   7.1 days

--------- Memory Report ---------
  layout:                 tp 1, pp 1, dp 256, ep 1
  floor per GPU:          131 GiB
    training state:       120 GiB
    activations:          3 GiB
    logits:               8 GiB
  overhead:               x2.00 (JAX-Titan Llama 3.1 8B pipeline)
  peak per GPU:           261 GiB of 238 GiB HBM
  fits:                   no (limit: 226 GiB, 95% of HBM)
  layout that fits:       tp 1, pp 1, dp 256, ep 1, fsdp: 23 GiB
  layout that fits:       tp 1, pp 2, dp 128, ep 1: 167 GiB
  layout that fits:       tp 2, pp 1, dp 128, ep 1: 145 GiB
  layout that fits:       tp 1, pp 2, dp 128, ep 1, fsdp: 49 GiB
  layout that fits:       tp 2, pp 1, dp 128, ep 1, fsdp: 27 GiB
  layout that fits:       tp 1, pp 4, dp 64, ep 1: 180 GiB

--------- Data Parallel Report ---------
  groups:                 32 scale-up domain(s) of 8 ranks
  all_reduce:             32.1 GB, ring/simple at 333 GB/s bus (model)
  time per step:          0.192 s against 1.613 s of compute per 4.19M-token step
  overlap:                hidden if overlapped
```

The default layout, plain data parallelism, keeps the whole training state on
every GPU (120 GiB for 8B parameters at 16 bytes each), so it does not fit;
the report lists layouts that do. Add `--fsdp`, `--tp` or `--pp` to plan one.
There is no XLA BF16 GEMM section: only the MI355X has a measured ceiling.

### Several GPU types: one row each

```bash
python -m jax_titan.planning.budget --model=llama3.1-70b --tokens=15e12 \
  --gpu=MI300X,MI325X,MI350X,MI355X,MI455X --gpus=2048 --tp=8 --fsdp
```

[Example: Llama 3.1 70B on 15T tokens, 2,048 GPUs](#example-llama-31-70b-on-15t-tokens-2048-gpus)
explains the communication column.

```text
llama3.1-70b, 15.00T tokens on 2048 GPUs (bf16)
GPU                     peak TF    MFU  tok/s/GPU         wall  GPU-hours     MWh         HBM GiB  DP comm/step  MFU source
---------------------------------------------------------------------------------------------------------------------------
AMD Instinct MI300X        1307  40.0%       1164    73.5 days      3.61M    2709       24/179 ok         0.89x  assumed
AMD Instinct MI325X        1307  40.0%       1164    73.5 days      3.61M    3613       24/238 ok         0.89x  assumed
AMD Instinct MI350X        2307  40.0%       2054    41.7 days      2.05M    2047       24/268 ok         1.57x  assumed
AMD Instinct MI355X        2517  40.0%       2241    38.2 days      1.88M    2628       24/268 ok         1.71x  assumed
AMD Instinct MI455X        5033  40.0%       4482    19.1 days    938.41k       -       24/402 ok         0.12x  assumed
HBM: the layout's floor x overhead over the GPU's HBM; DP comm/step: the data-parallel collectives over the step's compute, at --comm_efficiency of the links' peak (above 1x stays exposed even when overlapped).
```

### A time budget instead of tokens

`--days` (or `--gpu_hours`) gives the tokens the time buys.

```bash
python -m jax_titan.planning.budget --model=Qwen/Qwen3-235B-A22B --days=30 --gpu=MI355X \
  --gpus=1024 --ep=8 --fsdp
```

```text
--------- Model Overall Report ---------
  model:                  qwen3-235b-a22b
  config source:          Qwen/Qwen3-235B-A22B@8efa61729e24
  parameters:             235.1B
  active parameters:      22.2B
  FLOPs per token:        167.3 GFLOP at sequence 8192

--------- System Budget Full Report ---------
  ...
  tokens:                 15.876T (715 per active parameter)
  ...
  throughput per GPU:     6019 tokens/s
  wall time:              30.0 days
  ...
  GPU-hours:              737.28k
  ...

--------- Memory Report ---------
  layout:                 tp 1, pp 1, dp 1024, ep 8, fsdp
  ...
  peak per GPU:           17 GiB of 268 GiB HBM
  fits:                   yes (limit: 255 GiB, 95% of HBM)
  ...

--------- Data Parallel Report ---------
  groups:                 128 scale-up domain(s) of 8 ranks
  all_gather:             72.8 GB, ring/simple at 354 GB/s bus (model)
  all_gather:             72.8 GB, ring/simple at 354 GB/s bus (model)
  reduce_scatter:         145.5 GB, ring/simple at 357 GB/s bus (model)
  time per step:          0.819 s against 0.681 s of compute per 4.19M-token step
  overlap:                exposed; hiding it takes 5.04M tokens per step
```

Without `--ep=8 --fsdp` every GPU would hold the whole 235B-parameter state
(7,022 GiB) and nothing would fit. The 291 GB of FSDP traffic per step takes
longer than the step's compute; `--batch_tokens` of 5.04M or more would hide it.

### A dataset and epochs

```bash
python -m jax_titan.planning.budget --model=llama3.1-8b --dataset_tokens=1e12 --epochs=3 \
  --gpu=MI355X --gpus=256 --fsdp
```

```text
--------- System Budget Full Report ---------
  ...
  tokens:                 3.000T (374 per active parameter)
  ...
  throughput per GPU:     19558 tokens/s
  wall time:              7.0 days
  stepping time:          6.9 days
  GPU-hours:              42.73k
  energy:                 60 MWh at 100% of board power
  ...
```

### A model downloaded from Hugging Face

Planning needs only the model's `config.json`: download it (`hf download
Qwen/Qwen3-14B config.json --local-dir path/to/Qwen3-14B`), or save the raw file
from the model page's "Files and versions" tab into a directory. Pass the
directory or the file; `hf:` in front is optional. The report names the model
after the directory and gives the path as its source.

```bash
python -m jax_titan.planning.budget --model=path/to/Qwen3-14B --tokens=5e12 --gpu=MI355X \
  --gpus=256 --fsdp
```

```text
--------- Model Overall Report ---------
  model:                  Qwen3-14B
  config source:          path/to/Qwen3-14B/config.json
  parameters:             14.8B
  FLOPs per token:        94.0 GFLOP at sequence 8192

--------- System Budget Full Report ---------
  ...
  tokens:                 5.000T (339 per active parameter)
  ...
  wall time:              21.2 days
  ...
  GPU-hours:              130.07k
  ...

--------- Memory Report ---------
  ...
  peak per GPU:           29 GiB of 268 GiB HBM
  fits:                   yes (limit: 255 GiB, 95% of HBM)
  ...
```

### From what a JAX-Titan run measured

`--from_run` takes a JAX-Titan `--run_dir`, here the recorded
[Llama 3.1 8B recipe](recipes/maxtext-llama3.1-8b.md) run: its achieved FLOP/s
and power on the same GPU type (its MFU carried over to other types), its
checkpoint and restart costs, and its memory overhead replace the defaults.

```bash
python -m jax_titan.planning.budget --model=llama3.1-8b --tokens=2e12 --gpu=MI355X --gpus=512 \
  --seq_len=2048 --fsdp --from_run=docs/results/2026-11-07/maxtext-llama3.1-8b_recipe
```

```text
--------- System Budget Full Report ---------
  ...
  MFU:                    8.1% (measured on AMD Instinct MI355X)
  HFU:                    10.6%
  ETTR:                   0.985
  checkpoint interval:    1.46 h (Daly)
  ...
  throughput per GPU:     4351 tokens/s
  wall time:              10.5 days
  ...
  GPU-hours:              129.62k
  ...

--------- XLA BF16 GEMM Report ---------
  ...
  MFU of the ceiling:     14%

--------- Memory Report ---------
  ...
  overhead:               x2.83 (measured)
  peak per GPU:           8 GiB of 268 GiB HBM
  fits:                   yes (limit: 255 GiB, 95% of HBM)
  ...
```

## Models

`--model` takes any of these:

- A model in the model config index, by its name or its Hugging Face
  repository in any case: `llama3.1-70b`, `meta-llama/Llama-3.1-70B`,
  `Qwen3-235B-A22B`.
- A path to a model's `config.json` or the directory that holds it (`hf:` in
  front is optional): a model you downloaded. The path is all it needs; nothing
  has to be added to the index. A path that exists is read before any index
  name.
- `maxtext:<name>`: from the index when it has the name, otherwise from
  MaxText's config, which needs MaxText installed.
- A JAX-Titan Llama flavor, such as `llama3-test`.

All of them read Hugging Face `config.json` fields: Llama, Mistral, Mixtral,
Qwen and DeepSeek-style decoders, dense or mixture-of-experts. Multi-head
latent attention and sliding-window attention are not modeled. The report's
`config source` line says where the shape came from, and `--json` writes
everything the report prints.

### The model config index

The index, `src/jax_titan/planning/data/models/model_config_index.json`, keeps
the `config.json` fields planning reads, per model, with the repository and
commit they were read from
(`https://huggingface.co/<hf_config_source>/blob/<hf_config_revision>/config.json`).
Meta's repositories are gated, so the Llama entries come from ungated copies
(`mirror_of` names the original); they match MaxText's copies of Meta's
configs.

```bash
python -m jax_titan.planning.model_index
```

```text
llama3.1-405b      hugging-quants/Meta-Llama-3.1-405B-BNB-NF4-BF16@1f2a5deea6dc (mirror of meta-llama/Llama-3.1-405B)
llama3.1-70b       unsloth/Meta-Llama-3.1-70B@1b7306651142 (mirror of meta-llama/Llama-3.1-70B)
llama3.1-8b        unsloth/Meta-Llama-3.1-8B@e9a141a2091e (mirror of meta-llama/Llama-3.1-8B)
llama3.3-70b       unsloth/Llama-3.3-70B-Instruct@99cd0d2c829e (mirror of meta-llama/Llama-3.3-70B-Instruct)
mistral-7b         mistralai/Mistral-7B-v0.1@27d67f1b5f57
mixtral-8x22b      mistralai/Mixtral-8x22B-v0.1@e1cd34ff1747
mixtral-8x7b       mistralai/Mixtral-8x7B-v0.1@fc7ac94680e3
qwen3-0.6b         Qwen/Qwen3-0.6B@c1899de289a0
qwen3-1.7b         Qwen/Qwen3-1.7B@70d244cc86cc
qwen3-14b          Qwen/Qwen3-14B@40c069824f42
qwen3-235b-a22b    Qwen/Qwen3-235B-A22B@8efa61729e24
qwen3-30b-a3b      Qwen/Qwen3-30B-A3B@ad44e777bcd1
qwen3-32b          Qwen/Qwen3-32B@9216db5781bf
qwen3-4b           Qwen/Qwen3-4B@1cfa9a720891
qwen3-8b           Qwen/Qwen3-8B@b968826d9c46
```

### Adding a model to the index

Add a model when others should plan it by name, pinned to one commit. `--add
NAME REPO` takes a Hugging Face repository, not a local path: it reads that
repository's `config.json` from the Hub at its latest commit (or
`--revision`), keeps the fields planning reads, and records the repository and
commit. For a gated repository, set `HF_TOKEN`, or add an ungated copy with
`--mirror_of <original>`. Commit the changed index.

```bash
python -m jax_titan.planning.model_index --add qwen3-14b-base Qwen/Qwen3-14B-Base
```

```text
qwen3-14b-base: Qwen/Qwen3-14B-Base@0b0bd3732e2c374d483664439ea334928b65f304
```

Then `--model=qwen3-14b-base` and `--model=Qwen/Qwen3-14B-Base` both work.

## From a first estimate to a calibrated prediction

An assumed MFU answers whether a run is feasible. For a cluster size not yet
tested, the estimate has to say why the step takes the time it does: the part
that changes with scale (communication across nodes, collective tails,
overlap) is the part a single MFU hides. JAX-Titan works through these stages:

| Stage | Question | Tool |
|---|---|---|
| 1. Estimate | Is the run feasible in this allocation, time and budget? | `planning.budget` |
| 2. Benchmark the parts | Do kernels and collectives reach the rates the model assumes? | `observability.ceilings`, `scripts/bench_collectives.sh`, `planning.collectives` |
| 3. A short run on one node | Which part of the step differs from its prediction? | the trainer's `--run_dir` and `--profile_steps`, `planning.step --calibrate` |
| 4. Predict larger scales | What do the measured parts give at 64, 256 or 1,024 GPUs? | `planning.step --ladder` |
| 5. Scale up and check | Do the predictions hold at each rung? | the same runs, more nodes (Q1, Q2) |
| 6. Production | When will the run finish, and what changed when the estimate moves? | `planning.remaining`, `observability.perfdb` |

When a measurement disagrees with its prediction, correct the part
responsible (a kernel's rate, a collective's bandwidth, the overlap, the
runtime gaps) rather than the total, so the model still holds at other
scales and shapes. Hold the global batch and sequence length fixed for strong
scaling; grow the batch with the GPUs (`--weak`) as a separate experiment.

Hardware scaling says how fast the cluster runs the workload; it doesn't say
how many tokens the model needs. That comes from model-quality scaling:
smaller models over meaningful token budgets, and scaling laws. The time to
a target quality is the tokens it needs over the effective throughput.

## What it computes

**FLOPs.** Model FLOPs per token are three forward passes of the
parameters a token uses (the active experts in a mixture of experts), plus
causal attention, as in [metrics.md](metrics.md#definitions): the numerator of
MFU. Executed FLOPs add the forward pass rematerialization repeats (HFU).

`--attention` sets the attention convention:
- `causal` (the default) counts half of the attention scores, as JAX-Titan does.
- `none` gives the 6N convention of published compute figures, 7% less for a
  70B model at sequence 8,192.
- `full` counts the whole square, as PaLM and Megatron do.

`--tokens` counts every token processed, over all epochs.

**Time.** Wall time is tokens x FLOPs per token / (GPUs x dense peak at
`--precision` x MFU x ETTR). MFU is an input:

- `--mfu` (default 40%) is an assumption. The report prints the time at
  10-50% MFU, and on MI355X how the MFU compares with the BF16 GEMM rate XLA
  reaches (1,436 TFLOP/s, 57% of peak). 40% would be 70% of that.
- `--from_run` uses what a run measured. On the same GPU type the planner
  carries over the achieved FLOP/s; on another type, the MFU fraction, which is
  only an assumption there.

JAX-Titan's Llama 3.1 8B runs on 8x MI355X measure 8-12% MFU today
([pipelines.md](pipelines.md)).

**ETTR.** The planner projects ETTR for the GPU count with Daly's
checkpoint interval ([metrics.md](metrics.md)). The inputs are:

- the checkpoint cost (`--w_cp_s`, default the recipe's 2.4 s);
- the restart cost (`--restart_s`, default 175 s);
- the failure rate: Llama 3's 419 unexpected interruptions in 54 days on
  16,384 GPUs, 0.47 per 1,000 GPU-days.

**Energy.** Energy is GPUs x board power x `--power_fraction` x `--pue`
over the wall time, at the GPU boundary. AMD's board power is a maximum. The
recipe drew 48% of it, 670 W per MI355X, which `--from_run` picks up.

**Memory.** The analytic floor per GPU of the busiest stage has three parts:
- **training state:** `--state_bytes`, by default 4, 4 and 8 bytes per
  parameter of fp32 weights, gradients and AdamW moments;
- **stored activations:** the layer inputs with rematerialization, plus one
  layer recomputed at a time;
- **logits:** fp32 logits and their gradient.

The layout comes from `--tp`, `--pp`, `--ep`, `--fsdp`, `--batch_tokens`
and `--microbatches`. XLA's temporaries and unaliased buffers add a measured
factor on top:

| Run (8x MI355X, Llama 3.1 8B, 2 stages of 4) | Floor | Measured peak | Factor |
|---|---|---|---|
| JAX-Titan's model, gradients reduced per step | 92.7 GiB | 181-187 GiB | 1.96-2.01 |
| MaxText's layers, gradients reduced per microbatch | 62.8 GiB | 178 GiB | 2.83 |

The default is 2.0. `--from_run` measures the factor on the run, and
`memory.calibrate(run_dir)` gives it for any run with a memory event. The
report lists the layouts that fit, or the fewest GPUs that could hold the
state.

**Communication.** Each step, the data-parallel ranks all-reduce their
gradient; with FSDP they all-gather the weights twice and reduce-scatter the
gradient instead. Their times come from `planning.collectives`: measured on
the GPU where its table covers the group and size, and otherwise from its
model of RCCL (below).

- **Fully connected platforms:** on the MI300 and MI350 series, a collective
  over k of a platform's GPUs gets k-1 of each GPU's 7 links.
- **Efficiency:** where nothing is measured, links reach `--comm_efficiency`
  (0.75) of their peak, as RCCL does on MI355X.
- **Hidden or exposed:** the report compares the collectives with the step's
  compute and, when they're exposed, gives the batch that would hide them.

The planner doesn't model tensor, pipeline or expert-parallel communication,
or the pipeline bubble. For pipelines, `jax_titan.observability.flops` has
them per stage.

**Compute-optimal size.** For the same compute in Chinchilla's C = 6ND, at
20 tokens per parameter, the optimum is N = sqrt(C/120) and D = 20N.

## GPUs

Dense peaks (no structured sparsity) per GPU, from the vendors' pages,
retrieved 7 November 2026. `--gpu` also takes MI300, MI325, MI350, MI355 and
MI450 (AMD's MI450 series; its Helios GPU is the MI455X).

| GPU | BF16 TFLOP/s | FP8 | FP4 | HBM GB | TB/s | Board W | Scale-up GB/s per direction |
|---|---:|---:|---:|---:|---:|---:|---:|
| [MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) | 1,307.4 | 2,614.9 | | 192 | 5.3 | 750 | 448* (7 links, 8 GPUs) |
| [MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html) | 1,307.4 | 2,614.9 | | 256 | 6.0 | 1,000 | 448* (7 links, 8 GPUs) |
| [MI350X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html) | 2,306.9* | 4,613.7* | 9,227.5* | 288 | 8.0 | 1,000 | 504* (7 links, 8 GPUs) |
| [MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) | 2,516.6 | 5,033.2 | 10,066.3 | 288 | 8.0 | 1,400 | 504* (7 links, 8 GPUs) |
| [MI455X](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html) | 5,033.2* | 20,132.7* | 40,265.3* | 432 | 23.3 | | 1,800 (72-GPU rack) |
| [H100 SXM](https://www.nvidia.com/en-us/data-center/h100/) | 989.4* | 1,978.9* | | 80 | 3.35 | 700 | 450 |
| [H200 SXM](https://www.nvidia.com/en-us/data-center/h200/) | 989.4* | 1,978.9* | | 141 | 4.8 | 700 | 450 |
| [B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) | 2,250* | 4,500* | 9,000* | 180* | 8.0* | 1,000 | 900* |
| [TPU v5p](https://cloud.google.com/tpu/docs/v5p) | 459 | | | 95 | 2.765 | | |

\* Derived, not printed as such:
- **MI350X:** 256 compute units x 2.2 GHz x 4,096, 8,192 and 16,384 FLOPs per
  clock. The page rounds to 2.3, 4.6 and 9.2 PFLOP/s.
- **MI455X:** 256 work group processors x 2.4 GHz. AMD lists 5, 20.1 and 40.3
  PFLOP/s.
- **NVIDIA:** dense peaks are half the published sparse ones; B200 values are
  per-GPU shares of HGX and DGX B200 totals.
- **AMD scale-up:** links x per-link bandwidth.

**Two traps:**
- **NVIDIA's sparse peaks:** NVIDIA's pages print tensor peaks with sparsity,
  so using them overstates capacity 2x.
- **MI455X's FP8 ratio:** on MI455X, FP8 is 4x BF16, not 2x, so planning a
  BF16 run against the FP8 peak overstates capacity 4x.

Scale-out defaults to one 400 Gb/s NIC per GPU (50 GB/s per direction), and
300 GB/s on MI455X (600 GB/s both directions). `--scaleout_gbs` sets your
system's value.

The trainer's MFU uses AMD's rounded 2,500 TFLOP/s for MI355X
(`--peak_tflops_per_device`), 0.7% below the table's 2,516.6.
`--from_run` converts a run's MFU to achieved FLOP/s first, so the
difference doesn't carry over.

## Collectives: RCCL and MORI

`jax_titan.planning.collectives` gives one collective's time and bus
bandwidth by message size, with the algorithm and protocol the library picks,
for RCCL and for MORI-EP:

```bash
# A sweep of message sizes for one group: measured where the shipped table covers it.
python -m jax_titan.planning.collectives --op=all_reduce --ranks=8 --gpu=MI355X
python -m jax_titan.planning.collectives --op=all_reduce --ranks=1024 --nodes=128
python -m jax_titan.planning.collectives --op=dispatch --ranks=16 --nodes=2 --tokens=4096

# Measure this node (GPUs free), then turn the outputs into a table.
bash scripts/bench_collectives.sh ${JAX_TITAN_ROOT}/logs/collectives/today
python -m jax_titan.planning.collectives --calibrate ${JAX_TITAN_ROOT}/logs/collectives/today \
  --json table.json
```

**Sizes and bus bandwidth** follow rccl-tests' conventions. The message size S
is the buffer each rank contributes or receives in full. Bus bandwidth is S
over the time, times the operation's factor: 2(n-1)/n for all-reduce,
(n-1)/n for all-gather, reduce-scatter and all-to-all, and 1 for send/recv.
That makes it comparable with link bandwidth.

**The RCCL model** charges each collective a fixed latency plus its bytes on
the wire over the slowest path, for each algorithm and protocol, and takes
the fastest:
- **Algorithms:** ring for all-reduce, all-gather and reduce-scatter; tree
  for all-reduce; direct for all-to-all; one link or NIC for send/recv.
- **Protocols:** LL has low latency at 29% of the bandwidth; Simple has all
  of it after filling a pipeline on each step.
- **Bandwidth:** inside a node, the links among the group's GPUs (k-1 links
  for k GPUs of a fully connected MI300 or MI350 platform). Across nodes, a
  ring is also bounded by the group's NICs.
- **Latency across nodes:** as in NCCL's tuning model, a ring pays the
  inter-node hop on only the 2(N-1) of its steps that cross nodes, and a
  tree on 2 log2(N).

The constants come from a sweep on 8x MI355X: rccl-tests 2.18.3, RCCL 2.30.7,
one thread per GPU as JAX drives them. One thread for all the GPUs adds its
launches to every collective, which doubles the 8-GPU floor to 46 µs.

Each cell gives the latency floor and the peak bus bandwidth:

| Operation | 2 GPUs | 4 GPUs | 8 GPUs | RCCL's picks on 8 GPUs |
|---|---|---|---|---|
| All-reduce | 9.4 µs, 57.5 GB/s | 14.1 µs, 162 GB/s | 22.4 µs, 380 GB/s | Tree/LL to 4 KiB, Ring/LL to 1 MiB, then Ring/Simple on 112 channels |
| All-gather | 8.4 µs, 56.6 GB/s | 10.7 µs, 164 GB/s | 20.5 µs, 371 GB/s | Direct/Simple to 4 MiB, then Ring/Simple |
| Reduce-scatter | 8.3 µs, 56.1 GB/s | 11.1 µs, 160 GB/s | 25.0 µs, 370 GB/s | Ring/LL, then Ring/Simple |
| All-to-all | 8.5 µs, 55.9 GB/s | 9.1 µs, 149 GB/s | 25.3 µs, 335 GB/s | |
| Send/recv | 8.8 µs, 56.1 GB/s | 9.2 µs, 58.1 GB/s | 12.2 µs, 58.1 GB/s | one link |
| All-reduce, Tree forced | 8.8 µs, 27.8 GB/s | 12.8 µs, 61.3 GB/s | 21.2 µs, 126 GB/s | |

What these measurements mean:
- **Link efficiency:** collectives reach 74-80% of the Infinity Fabric
  links' peak (380 of 504 GB/s per direction for 8 GPUs).
- **Tree:** a forced tree gets a third of the ring's bandwidth.
- **Send/recv:** it gets one link, whatever the group.

From 64 MiB up, the model is within 3% of these measurements for all-reduce
and send/recv, and within 11% for the other operations. Below 4 MiB it's
mostly within 15%, but off by up to 1.5x at a few sizes around RCCL's protocol
switch. The measured points ship with JAX-Titan
(`planning/data/rccl_mi355x.json`, raw outputs in
`docs/results/2026-11-08/rccl-tests_mi355x/`), and predictions use them
wherever they cover the group and size.

They also match training: the recorded pipeline run's 15 GiB gradient
all-reduce over 4 GPUs takes 152 ms at the measured 158 GB/s. The profile
measured 139 ms of exposed collectives.

**MORI-EP** dispatch and combine send each token to the ranks of its top-k
experts. That's tokens x k x hidden x bytes per rank, the share for other
nodes over RDMA and the rest over XGMI. The model is MORI's published
bandwidth and latency (its README at `752efd37`) as a fixed latency plus the
slower share. That reproduces the published 128-token latencies:

| MI355X with AINIC | Dispatch XGMI / RDMA | Combine XGMI / RDMA | Latency at 128 tokens (dispatch / combine) |
|---|---|---|---|
| EP8 | 345 GB/s / - | 420 GB/s / - | 31 µs / 36 µs |
| EP16 | 179 / 54 GB/s | 234 / 71 GB/s | 84 µs / 108 µs |
| EP32 | 85 / 46 GB/s | 110 / 61 GB/s | 152 µs / 187 µs |

The bandwidth figures are for DeepSeek V3 shapes: 4,096 tokens, hidden
7,168, top-8, FP8 dispatch and BF16 combine. MI300X has its own published
numbers; MI325X and MI350X use their platform's. Other GPUs fall back to
RCCL's all-to-all. When MORI is installed, `scripts/bench_collectives.sh`
runs its EP benchmark too, and `--calibrate` reads the benchmark's
`MORI_PERF_OUT` records.

MORI also has JAX bindings (`mori.jax` registers a `mori_ep` FFI target) and
collectives (MORI-CCL, including an SDMA all-gather on the copy engines).
Those are for the MORI-EP milestone.

## Step time, calibration and the scale ladder

`jax_titan.planning.step` predicts one optimizer step on the busiest GPU as
compute + bubble + the exposed part of each collective + runtime gaps:

- **Compute:** the stage's executed FLOPs over the GEMM rate XLA reaches,
  1,436 TFLOP/s on MI355X (other GPUs get the same share of peak), times the
  share whole kernels reach.
- **Collectives:** one row per parallelism, from `planning.collectives`,
  with its library, operation, group, nodes, message size, calls per step,
  algorithm and bus bandwidth:
  - data-parallel gradient all-reduces;
  - FSDP's weight all-gathers and gradient reduce-scatters;
  - tensor-parallel all-reduces of activations;
  - pipeline sends and receives;
  - MORI-EP dispatch and combine.

  Each kind hides a share behind compute (`--overlap`).
- **Bubble:** (stages-1)/microbatches of the compute, a third of that for
  zero-bubble H1.
- **Runtime gaps:** measured idle time beyond the bubble.

```bash
python -m jax_titan.planning.step --model=llama3.1-8b --gpus=8 --pp=2 --seq_len=2048
python -m jax_titan.planning.step --calibrate runs/a --ladder 8,64,256,512 \
  --ladder_batch_tokens 4194304 --perfdb perf.jsonl
```

`--calibrate` rebuilds the layout a run used and sets each predicted part
against the run's profile (its busiest GPU). It then fits:
- the kernels' share of the GEMM rate;
- the collectives' hidden share, for the kinds that carried the run's traffic;
- the runtime gaps.

With that fit, the ladder predicts other GPU counts from measured parts
rather than from one MFU. For the recorded Llama 3.1 8B pipeline (8x MI355X,
2 stages of 4, 8 microbatches of 4 sequences of 2,048):

| Part | Predicted | Measured | Measured / predicted |
|---|---|---|---|
| Compute | 0.367 s | 1.122 s | 3.06 |
| Exposed collectives | 0.153 s | 0.139 s | 0.91 |
| Bubble and runtime gaps | 0.015 s | 0.448 s | 29.3 |
| Step | 0.535 s | 1.703 s | 3.19 |

The collectives are right. The gap is in compute and in runtime gaps:
- **Compute:** kernels reach 33% of the GEMM rate.
- **Runtime gaps:** the GPUs idle 0.40 s per step beyond the bubble. That's
  consistent with the step's 14.2k kernels (against 4.0k for FSDP) and the
  runtime's dispatch between them, but not yet broken down.

Those are what to fix, before more GPUs. With the fit and a 4M-token global
batch, the same two-stage shape scales as follows:

| GPUs | Nodes | Step | MFU | Tokens/s |
|---|---|---|---|---|
| 8 | 1 | 75.3 s | 12.9% | 55.7k |
| 64 | 8 | 9.83 s | 12.4% | 427k |
| 256 | 32 | 2.82 s | 10.8% | 1.49M |
| 512 | 64 | 1.65 s | 9.2% | 2.54M |

The data-parallel all-reduce across nodes and the fixed runtime gaps take a
growing share of each step as compute per GPU shrinks. At 1,024 GPUs a data-parallel
rank would get less than one sequence per microbatch, so the ladder stops.

## During a run: time left and the performance database

```bash
python -m jax_titan.planning.remaining runs/a --target_tokens 20e12
python -m jax_titan.observability.perfdb perf.jsonl
```

`remaining` divides the tokens left by the recent throughput (the last 200
steps) times the share of wall time spent on committed steps so far. That
share is ETTR so far: startup, saves, restarts and lost work.

Rerun it as the job runs. When the estimate moves, the run's records explain
why: MFU and step time over time, saves, restarts, power, and RCCL Inspector
times.

`step --calibrate --perfdb FILE` appends one record per run to the
performance database:
- **Key:** the model, GPU and layout.
- **Software:** packages and commits, the ROCm release, the driver and the
  GPU firmware.
- **Topology.**
- **Results:** the predicted and measured parts, and the part with the
  largest gap.

`perfdb` lists the records and flags a run that's slower than the best
earlier run with the same key: a regression after a change of XLA, RCCL, a
kernel provider, the driver or the firmware.

## Checked against

`tests/test_planning.py` and `tests/test_collectives.py` check the planner
against these numbers:

- **Llama 3 405B:** 405.9B parameters. On 15.6T tokens in the 6N convention
  that's 3.8e25 FLOPs, as in the Llama 3 paper.
- **Llama 3.1 405B's GPU-hours:** the model card's 30.84M H100-hours fall
  between the planner's figures at 43% MFU with ETTR 1 and at 38% MFU with
  ETTR 0.9. That range is the paper's MFU range (its Table 4).
- **PaLM 540B:** 238.3k tokens/s on 6,144 TPU v4 chips is 46.2% MFU with
  `--attention=full`, as published.
- **Mixtral 8x7B:** 46.7B parameters, 12.9B active.
- **Llama 3 70B on 20T tokens on one MI350X:** 286 years at 40% MFU with
  `--attention=none`; rounding the peak to 2.3 PFLOP/s and counting the
  embedding table gives 289. The state alone needs at least 8 GPUs.
- **The Scaling Book's Llama 3 70B example:** 15T tokens on four TPU v5p pods
  (35,840 chips) at 40% MFU take 11 days.
- **Memory:** the floor and factor of the recorded JAX-Titan run above.
- **The trainer's FLOPs:** the same FLOPs and rematerialization FLOPs per
  token as the trainer and `jax_titan.observability.flops` for Llama 3.1 8B.
- **RCCL:** the shipped MI355X measurements (380 GB/s all-reduce bus
  bandwidth on 8 GPUs) and the model within 10% of them at 256 MiB; the
  rccl-tests parser.
- **MORI-EP:** the published 31, 84 and 152 µs dispatch latencies for EP8,
  EP16 and EP32.
- **The step model:** the recorded run's gradient all-reduce message and
  exposed collective time; a calibration that reproduces its 1.703 s step;
  the ladder's strong and weak scaling.
- **Time left and the performance database:** a synthetic run, and a flagged
  regression.

## Example: Llama 3.1 70B on 15T tokens, 2,048 GPUs

The run uses tensor parallelism 8 inside each 8-GPU platform and FSDP across
platforms, with sequence 8,192, 4M-token steps, 40% MFU assumed and no power
measurement:

```text
GPU                     peak TF    MFU  tok/s/GPU         wall  GPU-hours     MWh         HBM GiB  DP comm/step
AMD Instinct MI300X        1307  40.0%       1164    73.5 days      3.61M    2709       24/179 ok         0.89x
AMD Instinct MI325X        1307  40.0%       1164    73.5 days      3.61M    3613       24/238 ok         0.89x
AMD Instinct MI350X        2307  40.0%       2054    41.7 days      2.05M    2047       24/268 ok         1.57x
AMD Instinct MI355X        2517  40.0%       2241    38.2 days      1.88M    2628       24/268 ok         1.71x
AMD Instinct MI455X        5033  40.0%       4482    19.1 days    938.41k       -       24/402 ok         0.12x
```

The MI300X and MI325X have the same peak, so they take the same time; the
MI325X's higher board power costs more energy at full power.

The communication column compares FSDP's collectives with the step's compute:
- **MI300X and MI325X:** over one 400 Gb/s NIC per GPU, the collectives take
  0.89x the compute, so they can hide.
- **MI350X and MI355X:** they take 1.6-1.7x, because those parts compute twice
  as fast. That traffic stays exposed even with full overlap. Those GPUs need
  a larger batch, more NIC bandwidth, or less data-parallel traffic (fewer
  data-parallel ranks, or reduction in bf16 with `--grad_reduce_bytes=2`).
- **MI455X:** its 72-GPU racks and 300 GB/s of scale-out hide it.

The collectives across nodes come from `planning.collectives`' model (below),
whose constants for crossing nodes are still assumptions.

## Cross-check: BBuf's simulator

BBuf's [model-compute-simulation](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/skills/model-compute-simulation/SKILL.md)
skill is an inference simulator. It counts per-op forward FLOPs for a
serving shape, with no backward pass, optimizer, memory, communication or
time to train. Its repository has no license, so none of its code or data
is used here, only cross-checked. Its forward FLOPs agree with
`jax_titan.observability.flops` for matmuls, causal attention (2 x heads x
head dimension x sequence per token and layer), the router and the routed
experts. It differs in two ways: it also counts elementwise ops, and by
default it computes the logits of the last token only, as in inference.

## Limitations

- **Time to quality:** the planner doesn't predict it. The tokens needed for
  a target loss come from scaling laws or smaller runs, and
  `summarize --target_loss` measures a run's time to target.
- **MFU:** it's an input, and only measured on MI355X so far. Planning other
  GPU types from it, or at the default 40%, is an assumption.
- **Memory factor:** it was measured on two 8B pipelines on one node.
  FSDP, tensor-parallel and multi-node layouts are unvalidated, and for FSDP
  the factor applies to a small floor.
- **Communication in the budget planner:** it covers data parallelism and
  FSDP only. `planning.step` covers tensor, pipeline and expert parallelism
  too, but not context parallelism.
- **Across nodes:** the collective constants (5 µs per inter-node step, NICs
  at 90%) are assumptions until a multi-node sweep (Q1) measures them. That
  sweep needs rccl-tests built with MPI.
- **MORI:** it isn't installed here, so MORI-EP is modeled from its published
  numbers.
- **The step model's compute:** it's one efficiency for every kernel. The next
  refinement is per-op: GEMM shapes against the roofline and the measured
  ceilings, and attention microbenchmarks.
- **The step model's overlap:** it's fitted from one run's exposed collective
  time.
- **Multi-head latent attention** (DeepSeek V2 and V3, Kimi K2) isn't
  modeled.
