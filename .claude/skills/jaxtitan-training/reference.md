# JAX-Titan Training Reference

## Run Names and Files

Name a run `{model}_{layout}_{schedule}[_{variant}]`, for example
`llama3.1-8b_pp2_1f1b` or `llama3.1-8b_fsdp8`. Per run:

| File | Content |
|------|---------|
| `${JAXTITAN_ROOT}/logs/<topic>/<run>.out` | stdout and stderr, including compile time and peak HBM |
| `${JAXTITAN_ROOT}/logs/<topic>/<run>.jsonl` | `--log_file`: one record per step (loss, step time, tokens/s, MFU) |
| `docs/results/<run>.jsonl` | Logs kept as evidence for a documented result |

## Per-Run Record

Record these for every result you report; a comparison is only valid when
they match or the difference is the thing being measured:

- commit of the repo, and the jax, XLA and Shardy revisions with their patches
- the exact command, including `XLA_FLAGS`
- model, tokens per step (microbatches x microbatch size x sequence length)
- stages, schedule, gradient-reduction mode
- `--steps` (it sets the learning-rate decay) and `--seed`
- median step time over steps 5 to N, tokens/s, MFU, peak HBM range, compile
  time, step-0 loss
- whether another user's process was on the GPUs during the run

## Notes File

`${JAXTITAN_ROOT}/notes/debug-log.md` is the working memory across sessions.
Read all of it at the start of a new debug session; append after each
meaningful test. A new chat, a new day, or returning after unrelated work is
a new session. Each entry:

```markdown
## 2026-11-05 RCCL hang at step 0
- Symptom: no step 0 after compile; rendezvous leader stuck in Initialize clique
- Suspicion: another user's job started on the same GPUs
- Evidence: two hung attempts while another user's job ran; the rerun on an idle node passed
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

Run the same layout with 4 and 8 microbatches at the same microbatch size.
Step time is a fixed cost plus a per-microbatch time; the fixed cost holds
the bubble and the once-per-step work, such as the gradient all-reduce and
the optimizer update. With 2 stages, 1F1B's theoretical bubble is
1/(microbatches + 1) of the step.
