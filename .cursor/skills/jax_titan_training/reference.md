# JAX-Titan Training Reference

## Run Names and Files

Name a run `{model}_{layout}_{schedule}[_{variant}]`, for example
`llama3.1-8b_pp2_zero-bubble_fsdp` or `llama3.1-8b_fsdp8`. Per run:

| File | Content |
|------|---------|
| `${JAX_TITAN_ROOT}/logs/<topic>/<run>.out` | stdout and stderr, including compile time and peak HBM |
| `${JAX_TITAN_ROOT}/logs/<topic>/<run>.jsonl` | `--log_file`: one record per step (loss, step time, tokens/s, MFU) |
| `--profile_dir` | `jax.profiler` traces; summarize with `python -m jax_titan.observability.profile DIR` |
| `${JAX_TITAN_ROOT}/logs/<topic>/<run>.hlo` | XLA dump: `XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS --xla_dump_to=DIR --xla_dump_hlo_as_text"` with `JAX_ENABLE_COMPILATION_CACHE=false` (a cache hit dumps nothing); its GEMMs per step, with library and kernel configuration, with `python3 src/jax_titan/observability/gemms.py DIR --run_dir RUN --merge [--sites]`, which needs only the Python standard library |
| `docs/results/<date>/<run>.jsonl` | Logs kept as evidence for a documented result |

## Per-Run Record

Record these for every result you report; a comparison is only valid when
they match or the difference is the thing being measured:

- commit of the repo, and the jax, XLA and Shardy revisions with their patches
- the exact command, including `XLA_FLAGS`
- model, tokens per step (microbatches x microbatch size x sequence length)
- stages, schedule, layout inside each stage, gradient-reduction mode
- `--steps` (it sets the learning-rate decay) and `--seed`
- median step time over steps 5 to N, tokens/s, MFU, peak HBM range, compile
  time, step-0 loss
- whether another user's process was on the GPUs during the run

## Notes File

`${JAX_TITAN_ROOT}/notes/debug-log.md` is the working memory across sessions.
Read all of it at the start of a new debug session; append after each
meaningful test. A new chat, a new day, or returning after unrelated work is
a new session. Each entry:

```markdown
## 2026-11-05 RCCL hang at step 0
- Symptom: no step 0 after compile; rendezvous leader stuck in Initialize clique
- Suspicion: another user's job started on the same GPUs
- Evidence: two hung attempts while root's minigpt_profile.py ran; the rerun on an idle node passed
- Next check: none
- Status: resolved (run only on an idle node)
```

Log negative results that rule a suspicion out; do not log identical reruns.

## Stopping a Bad Run

Stop a run instead of letting it continue "for more signal" when:

- it shows a red flag from SKILL.md (hang at step 0, out of memory, a
  non-finite loss);
- step time drifts more than 5% from the same layout's earlier median;
- another user's process appears on its GPUs, since its numbers no longer
  count.

Then follow the cleanup checklist in SKILL.md, keeping the logs.

## Measuring the Bubble

Two ways, both used for `docs/pipelines.md`:

- From step times: run the same layout with 4 and 8 microbatches. Step time
  is a fixed cost plus a per-microbatch time; the fixed cost holds the
  bubble and the once-per-step work.
- From traces: `--profile_steps=20:22`, then
  `python -m jax_titan.observability.profile DIR` gives each GPU's compute,
  collective, copy and idle shares. Discard runs whose first traced step was
  much slower than the untraced ones.
