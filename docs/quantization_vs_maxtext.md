# Quantized recipes: JAX-Titan against MaxText (2026-11-12, 8x MI350X)

JAX-Titan's `xla_fp8`, `xla_mxfp8` and `xla_mxfp4` providers against the
recipes of MaxText they follow, on Llama 3.1 8B: throughput on one GPU and
with FSDP over 8 GPUs, and the loss of a 500-step training from scratch. The
JAX-Titan numbers are new runs, with rows in
[results/2026-11-12](results/2026-11-12) and
[results/2026-11-11](results/2026-11-11). The MaxText numbers are earlier
measurements on the same node and were not run again.

## Summary

- **Throughput.** BF16 in JAX-Titan reaches 0.96x of MaxText on one GPU and
  0.90x with FSDP over 8. The quantized recipes keep their order: scalar FP8
  is the fastest in both frameworks (1.15x to 1.19x of BF16 in JAX-Titan,
  1.22x to 1.24x in MaxText), and the MX recipes are within 0.91x to 1.05x of
  BF16 in both.
- **Loss.** All four recipes train to the end in both frameworks without a
  non-finite step. MXFP4 ends about 1% above BF16 in both (+1.05% here,
  +1.11% in MaxText). MXFP8 ends +0.86% above BF16 here against +0.11% in
  MaxText, and scalar FP8 +2.36% against +0.28%.
- **The loss comparison is weaker than the throughput comparison.** The one
  BF16 run with another seed ends 1.27% from the first, against 0.13% in
  MaxText, so only the FP8 gap is clearly beyond what another seed does here.

## What is compared

| | JAX-Titan | MaxText, throughput | MaxText, loss |
|---|---|---|---|
| Measured | 2026-11-11 and 12 | 2026-11-02 and 03 | 2026-10-14 and 15 |
| XLA | openxla `d23c30982a`, no patch | openxla `f5bbeeb706` (2026-11-01) with the two hipBLASLt scale-layout commits, the flag at its default (natural layout) | openxla main of 2026-09-29 with the scale-layout commits and the multi-GPU composite fix, natural layout |
| JAX | `6b39d495e5` | `5d180d800e` (2026-11-02) | main of 2026-09-29 with the MXFP4 recipe |
| ROCm | 10.1.0 (TheRock), hipBLASLt 1.4.1 | 7.14.0 (TheRock), hipBLASLt 1.4 | 7.14.0 |
| Model | JAX-Titan's Llama 3.1 8B, vocabulary 128,256 | MaxText Llama3-8B | MaxText Llama3-8B, vocabulary 151,936 |
| Data | WikiText-103, Llama 3 tokenizer | synthetic | C4, Qwen3 tokenizer |
| BF16 | native JAX | no quantization | the same |
| Scalar FP8 | `xla_fp8`: per-tensor, scale from the current tensor | `quantization=fp8`: per-tensor, delayed scaling over the last 1024 maxima | the same |
| MXFP8, MXFP4 | `xla_mxfp8`, `xla_mxfp4`: `jax.nn.scaled_dot_general` | `jax_mxfp8`, `jax_mxfp4`: the same call and configuration | the same |

No XLA in this comparison has the swizzled scale layout chosen or any later
MX performance work: every hipBLASLt MX GEMM takes the natural scale layout.
The scalar FP8 recipes differ by design. Every recipe quantizes the same
GEMMs in both frameworks: the attention q/k/v/o projections and the three
GEMMs of the MLP.

## Throughput

Sequence 2,048, 8 sequences per GPU and step (16,384 tokens). TPS is tokens
per second per GPU over the median of 20 steps after 8 warm-up steps, the
protocol of both sides. JAX-Titan's FSDP numbers are the median of three runs,
the range in parentheses; every other number is one run.

| Recipe | 1 GPU: JAX-Titan | MaxText | JAX-Titan / MaxText | FSDP over 8: JAX-Titan | MaxText | JAX-Titan / MaxText |
|---|---|---|---|---|---|---|
| BF16 | 10,443 | 10,844 | 0.963 | 9,780 (9,768 to 9,786) | 10,884 | 0.899 |
| Scalar FP8 | 12,452 | 13,425 | 0.928 | 11,207 (11,203 to 11,208) | 13,233 | 0.847 |
| MXFP8 | 9,862 | 10,508 | 0.939 | 9,582 (9,581 to 9,594) | 10,014 | 0.957 |
| MXFP4 | 9,549 | 10,094 | 0.946 | 10,132 (9,938 to 10,238) | 11,405 | 0.888 |

Against the BF16 of the same framework:

| Recipe | 1 GPU: JAX-Titan | MaxText | FSDP over 8: JAX-Titan | MaxText |
|---|---|---|---|---|
| Scalar FP8 | 1.192 | 1.235 | 1.146 | 1.216 |
| MXFP8 | 0.944 | 0.967 | 0.980 | 0.920 |
| MXFP4 | 0.914 | 0.929 | 1.036 | 1.048 |

- **BF16 is the larger part of the difference.** JAX-Titan's BF16 is 4% below
  MaxText on one GPU and 10% below with FSDP over 8, before any recipe is
  involved. Model code, ROCm release and XLA revision all differ; this
  comparison does not separate them.
- **The recipes cost about the same in both.** On one GPU each quantized
  recipe is 2% to 4% further from BF16 in JAX-Titan than in MaxText. With FSDP
  over 8, MXFP4 is within 1% of MaxText's ratio, MXFP8 is 6% better and scalar
  FP8 6% worse.
- **Run-to-run spread** of JAX-Titan with FSDP over 8 is 0.2% for BF16, FP8
  and MXFP8 and 3.0% for MXFP4, whose second run had one slow step inside the
  window.
- **The first step of an MX run is slow on this XLA.** With FSDP over 8,
  MXFP8 needs 320 s to compile and 480 to 490 s for its first step, MXFP4
  1,250 s and 2,160 to 2,210 s, in all three runs; BF16 and FP8 need 213 s
  and 12 to 14 s. On one GPU the first step takes 2 s for every recipe and
  MXFP4 compiles in 1,250 s. MaxText recorded the same behaviour for the
  natural scale layout.

## Loss

Llama 3.1 8B from scratch with FSDP over 8 GPUs and the hyperparameters of
MaxText's study: 2 sequences of 2,048 tokens per GPU and step (32,768 tokens),
500 steps, AdamW (beta2 0.95, weight decay 0.1), peak learning rate 1e-4 after
50 warm-up steps, cosine decay to 10%, gradient clipping at 1.0. All recipes
of a framework start from the same weights and read the same batches. The
final loss is the mean training loss of the last 50 steps, each computed
before its update on a batch not seen before. JAX-Titan also reports the loss
on 4 held-out batches of WikiText-103 validation after step 500.

| Recipe | JAX-Titan: first loss | Final loss | vs BF16 | Validation loss | vs BF16 | MaxText: first loss | Final loss | vs BF16 |
|---|---|---|---|---|---|---|---|---|
| BF16 | 12.5930 | 4.8780 | | 4.8148 | | 12.4410 | 5.4456 | |
| Scalar FP8 | 12.5937 | 4.9929 | +2.36% | 4.9253 | +2.30% | 12.4458 | 5.4607 | +0.28% |
| MXFP8 | 12.5894 | 4.9201 | +0.86% | 4.8486 | +0.70% | 12.4412 | 5.4515 | +0.11% |
| MXFP4 | 12.5802 | 4.9294 | +1.05% | 4.8690 | +1.13% | 12.4419 | 5.5058 | +1.11% |
| BF16, other seed | 12.5839 | 4.9398 | +1.27% | 4.7814 | -0.69% | 12.4329 | 5.4528 | +0.13% |

Losses of the two frameworks are not comparable in absolute terms: data,
tokenizer and vocabulary differ. The columns to compare are the gaps to BF16.

- **MXFP4 agrees.** About 1% above BF16 in both frameworks, in the training
  loss and in JAX-Titan's validation loss.
- **MXFP8 is further from BF16 here**, +0.86% against +0.11%, but inside the
  spread of JAX-Titan's seed run.
- **Scalar FP8 is the largest gap here**, +2.36% and +2.30%, against +0.28% in
  MaxText. The recipes differ: JAX-Titan scales each tensor by its current
  maximum, MaxText by a running maximum. This comparison does not show whether
  the scaling rule is the cause.
- **The seed run is a weak noise band.** In JAX-Titan the seed changes the
  initial weights, the training batches and the validation batches, so its
  final losses are on other data than the first run's; in MaxText it changes
  only the weights. It shows that 1% is not resolved by one run per recipe.
- **The start of training is rougher in JAX-Titan.** Every run but MXFP8 has a
  loss peak of 14 to 17 between steps 16 and 24, and the gradient norm before
  clipping reaches 158 to 609 in the first 50 steps, against 14 to 22 in
  MaxText. After step 50 no run has a gradient norm above 10 except one step of
  the seed run.

## GEMMs

Static counts in XLA's optimized HLO of the FSDP training step, 16,384 tokens
per GPU:

| Recipe | On quantized operands | On operands dequantized from MX data | Not quantized |
|---|---|---|---|
| BF16 | 0 | 0 | 35 |
| Scalar FP8 | 27 | 0 | 11 |
| MXFP8 | 26 | 21 | 17 |
| MXFP4 | 47 | 0 | 17 |

Every projection and MLP GEMM of a quantized recipe runs on quantized data or,
for part of MXFP8, on BF16 data dequantized from it, which is the autotuner's
choice by measured time. MaxText's programs have the same structure: 27 FP8
GEMMs, and 47 quantized GEMMs for the MX recipes, of which MXFP8 runs 24
dequantized.

## Limits

- One run per configuration except JAX-Titan's FSDP throughput.
- XLA, JAX and ROCm are close, not equal; the MaxText runs are 9 and 28 days
  older than the JAX-Titan runs.
- MaxText's throughput runs use synthetic data, which does not change the
  shapes or the kernels.
- Training loss on unseen batches is the only metric shared by both sides;
  there is no downstream evaluation.
- On this XLA the natural scale layout of hipBLASLt is the only one. MaxText's
  measurements found its MXFP4 result to deviate at the fourth digit of the
  first loss from the swizzled layout and from the Triton kernel.

## Reproduction

```bash
# Throughput, one GPU and FSDP over 8 (add --ops for a quantized recipe)
python -m jax_titan.trainer.train --parallel=dp --microbatches=1 --microbatch_size=8 --steps=40
python -m jax_titan.trainer.train --parallel=fsdp --microbatches=8 --microbatch_size=8 --steps=40 \
    --ops=linear=xla_mxfp8,swiglu=xla_mxfp8

# Loss
python -m jax_titan.trainer.train --parallel=fsdp --microbatches=8 --microbatch_size=2 --steps=500 \
    --lr=1e-4 --warmup_steps=50 --eval_every=100 --ops=linear=xla_mxfp4,swiglu=xla_mxfp4
```

MX runs with FSDP need `--step_timeout=0` and, in `XLA_FLAGS`,
`--xla_gpu_executable_terminate_timeout` and
`--xla_gpu_first_collective_call_terminate_timeout_seconds` above the first
step's duration.
