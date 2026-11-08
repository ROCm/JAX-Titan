# Planning a training run

`jaxtitan.planning.budget` estimates a training run before it starts: its
FLOPs, time, GPU-hours, energy and cost, whether its layout fits in memory,
and whether its data-parallel communication hides behind compute. Give it a
model, a token or time budget, and one GPU type or several to compare.

```bash
# Tokens to train on GPUs of one type: the full report.
python -m jaxtitan.planning.budget --model=llama3.1-8b --tokens=2e12 --gpu=MI325X --gpus=256

# The same run on several GPU types: one row each.
python -m jaxtitan.planning.budget --model=maxtext:llama3.1-70b --tokens=15e12 \
  --gpu=MI300X,MI325X,MI350X,MI355X,MI455X --gpus=2048 --tp=8 --fsdp

# A time budget instead (--days or --gpu_hours), or a dataset and epochs.
python -m jaxtitan.planning.budget --model=hf:path/to/config.json --days=30 --gpu=MI355X --gpus=1024
python -m jaxtitan.planning.budget --model=llama3.1-8b --dataset_tokens=1e12 --epochs=3 ...

# From what a JAX-Titan run measured: its achieved FLOP/s and power on the same GPU type
# (its MFU carried over to other types), checkpoint and restart costs, and memory overhead.
python -m jaxtitan.planning.budget --model=maxtext:llama3.1-8b --tokens=2e12 --gpu=MI355X \
  --gpus=512 --seq_len=2048 --from_run=runs/maxtext-llama3.1-8b
```

`--model` is a JAX-Titan Llama flavor, `maxtext:<name>` or
`hf:<config.json>` (Llama, Mistral, Mixtral, Qwen and DeepSeek-style
decoders; multi-head latent attention is not modeled). `--json` writes
everything the report prints.

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
gradient instead. The collectives are hierarchical: first inside a scale-up
domain over the GPU links, then across domains over each GPU's own NIC.

- **Fully connected platforms:** on the MI300 and MI350 series, a collective
  over k of a platform's GPUs gets k-1 of each GPU's 7 links.
- **Efficiency:** collectives reach `--comm_efficiency` (0.7) of the links'
  peak. JAX-Titan's RCCL Inspector measured 57-79% for 4-GPU all-reduces on
  MI355X.
- **Hidden or exposed:** the report compares the collectives with the step's
  compute and, when they're exposed, gives the batch that would hide them.

The planner doesn't model tensor, pipeline or expert-parallel communication,
or the pipeline bubble. For pipelines, `jaxtitan.observability.flops` has
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

## Checked against

`tests/test_planning.py` checks the planner against these numbers:

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
  token as the trainer and `jaxtitan.observability.flops` for Llama 3.1 8B.

## Example: Llama 3.1 70B on 15T tokens, 2,048 GPUs

The run uses tensor parallelism 8 inside each 8-GPU platform and FSDP across
platforms, with sequence 8,192, 4M-token steps, 40% MFU assumed and no power
measurement:

```text
GPU                     peak TF    MFU  tok/s/GPU         wall  GPU-hours     MWh         HBM GiB  DP comm/step
AMD Instinct MI300X        1307  40.0%       1164    73.5 days      3.61M    2709       24/179 ok         1.14x
AMD Instinct MI325X        1307  40.0%       1164    73.5 days      3.61M    3613       24/238 ok         1.14x
AMD Instinct MI350X        2307  40.0%       2054    41.7 days      2.05M    2047       24/268 ok         2.01x
AMD Instinct MI355X        2517  40.0%       2241    38.2 days      1.88M    2628       24/268 ok         2.20x
AMD Instinct MI455X        5033  40.0%       4482    19.1 days    938.41k       -       24/402 ok         0.19x
```

The MI300X and MI325X have the same peak, so they take the same time; the
MI325X's higher board power costs more energy at full power. Over one 400 Gb/s
NIC per GPU, FSDP's collectives take longer than the step's compute on every
MI300 and MI350 part. That traffic stays exposed even with full overlap, so
those GPUs need a larger batch, more NIC bandwidth, or less data-parallel
traffic (fewer data-parallel ranks, or reduction in bf16 with
`--grad_reduce_bytes=2`). The MI455X's 72-GPU racks and 300 GB/s of
scale-out hide it.

## Cross-check: BBuf's simulator

BBuf's [model-compute-simulation](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/skills/model-compute-simulation/SKILL.md)
skill is an inference simulator. It counts per-op forward FLOPs for a
serving shape, with no backward pass, optimizer, memory, communication or
time to train. Its repository has no license, so none of its code or data
is used here, only cross-checked. Its forward FLOPs agree with
`jaxtitan.observability.flops` for matmuls, causal attention (2 x heads x
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
- **Communication:** the model covers data parallelism and FSDP only.
- **Multi-head latent attention** (DeepSeek V2 and V3, Kimi K2) isn't
  modeled.
