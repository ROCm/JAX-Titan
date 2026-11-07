# Recipe: MaxText Llama 3.1 8B as a 2-stage pipeline on 8x MI355X

JAX-Titan's M4 demo: MaxText 0.2.5's Llama 3.1 8B layers as a Shardy MPMD
pipeline with ROCm TE fused attention, trained on WikiText-103 with every
metric of [metrics.md](../metrics.md) recorded.

```bash
export JAXTITAN_LAUNCH_TIME=$(date +%s.%N)
XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS" python -m jaxtitan.trainer.train \
  --model=maxtext:llama3.1-8b --num_stages=2 --microbatches=8 --microbatch_size=4 \
  --seq_len=2048 --ops=attention=te --steps=1000 --warmup_steps=100 \
  --eval_every=100 --eval_batches=4 \
  --checkpoint_dir=${JAXTITAN_ROOT}/checkpoints/maxtext-llama3.1-8b --checkpoint_every=250 \
  --run_dir=runs/maxtext-llama3.1-8b
python -m jaxtitan.observability.summarize runs/maxtext-llama3.1-8b --wcp_steps 50
python -m jaxtitan.observability.ettr runs/maxtext-llama3.1-8b --wcp_steps 50
```

Add `--init_from=<converted checkpoint>` to start from Meta's weights instead
of MaxText's random initialization ([getting_started.md](../getting_started.md)
has the conversion).

**What runs.** Two stages of 4 GPUs with 16 decoder layers each, the
zero-bubble H1 schedule, 8 microbatches of 4 sequences of 2,048 tokens
(65,536 tokens per step), data parallelism inside each stage with gradients
all-reduced in every microbatch's backward pass. Attention is TE's fused
attention (AITER/CK FMHA) through `jaxtitan.ops`; everything else is
MaxText's code in native JAX. AdamW (3e-4 after 100 warmup steps, cosine
decay to 10%), one gradient norm over both stages clipped at 1.0, bf16
compute with fp32 weights and optimizer state, full rematerialization.
Validation runs on 4 fixed batches of WikiText-103's validation split.

**Stack.** JAX-Titan `ea10eb6`; jax 0.12.1 with JAX-Titan's jaxlib and Shardy
patches; TheRock ROCm 10.1 container; the ROCm TE nightly wheel of
2026-11-02; MaxText 0.2.5 at `493d59281`.

## Expected results (2026-11-07)

| Metric | Value |
|---|---|
| Step time | 1.857 s, median of three 30-step runs (spread 0.4%); regression threshold 1.866 s. Over the 1,000-step run: p50 1.886 s, p95 1.923 s |
| Throughput | 35.3k tokens/s, 4.4k per GPU |
| MFU | 8.2% of the nominal 2,500 TFLOP/s per GPU; 14.3% of the 1,436 TFLOP/s BF16 GEMM rate XLA reaches |
| Peak HBM | 177-178 GiB per GPU |
| Startup | 73 s cold: compile 40 s, the first step's excess over a steady step 17 s, init 9 s, setup 5 s, JAX init 2 s |
| Restart | 98 s from a checkpoint: restore 31 s, compile 43 s, first step 17 s |
| Checkpoint | 96 GB of weights and AdamW state; a save blocks the step loop 1.1 s, and the next 50 steps take 1.2-1.4 s longer in total while Orbax writes. The first save of a process blocks 77-78 s |
| Energy | 0.165-0.178 J/token at the GPUs (5.4-5.8 kW for the 8 GPUs, GFX clocks 2.16-2.25 GHz) |
| Gate 0 | The pipeline stays within 8.7e-4 relative loss of `--parallel=dp` over the first 10 steps with the default warmup ([pipelines.md](../pipelines.md#global-gradient-clipping-maxtext-attention-and-gate-0-2026-11-07-8x-mi355x)) |

Reference loss curve, for regression checks (training loss of the step, and
validation loss after it):

| Step | 0 | 99 | 199 | 299 | 399 | 499 | 599 | 699 | 799 | 899 | 999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Training loss | 12.30 | 6.02 | 5.21 | 5.47 | 5.38 | 5.48 | 5.23 | 5.21 | 5.00 | 4.88 | 4.84 |
| Validation loss | | 5.90 | 5.12 | 5.38 | 5.33 | 5.32 | 5.23 | 5.10 | 4.95 | 4.84 | 4.75 |

The loss plateaus between steps 200 and 600 with gradient norms of 30-76
(clipped to 1.0); the learning rate is not tuned. The validation loss reaches
5.0 at step 800.

**Restart rehearsal.** The run above was made with `--fail_at_step=620`,
which exits abruptly after step 619, and then rerun unchanged: it restored
step 500 and finished. Over both attempts (2,552 s), ETTR was 0.739, or
25.7k effective tokens/s against 34.7k steady. The rest was startup 171 s
(both attempts), checkpoints 159 s (mostly the two first saves), 120 lost
steps (225 s), validation 23 s, and 89 s of other time, 49 s of it between
the crash and the relaunch. With a steady checkpoint cost of 2.4 s and a
restart cost of 175 s (the restart plus a first save), Daly's interval is
1.0 h at 8 nodes for 3.8 failures per 1,000 node-days (projected ETTR
0.999), and 3.6 minutes at 2,048 nodes (0.964; 0.885 at 17.5 failures per
1,000 node-days). Records are in
[results/2026-11-07/maxtext-llama3.1-8b_recipe](../results/2026-11-07/maxtext-llama3.1-8b_recipe).

## Known limits

- Gradients are all-reduced in every microbatch: MaxText's own trainer runs
  its pipeline at 1.763 s per step with MaxText's attention, against 2.098 s
  here with the same attention. JAX-Titan's Llama, which reduces once per step,
  takes 1.502 s. Per-step reduction for MaxText's layers is the next step.
- One node; multi-node is M5.
