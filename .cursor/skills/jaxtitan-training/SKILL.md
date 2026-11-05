---
name: jaxtitan-training
description: >-
  Runs, monitors, compares and cleans up JAX-Titan training jobs on AMD
  Instinct GPUs. Use when launching or benchmarking jaxtitan.trainer.train or
  the examples, when a run hangs, slows down, runs out of memory or diverges,
  when comparing the pipeline against the no-pipeline baseline, and after a
  job finishes, fails or is stopped, to stop its processes, free the GPUs and
  remove leftover files.
---

# JAX-Titan Training Guide

## Core Principle

GPU time on a shared node is the scarce resource. Start a run only on an idle
node, put a timeout on every run, judge every result against a baseline
measured on the same stack, and free the GPUs as soon as a job is over. Keep
notes in `${JAXTITAN_ROOT}/notes/debug-log.md` (`JAXTITAN_ROOT` defaults to
`.jaxtitan` next to the repo); read it at the start of every new debug
session.

## Before a Run

1. On the host, `scripts/gpus_free.sh` must print `free`. If it prints
   `busy`, wait or ask the user. Never stop another user's process without
   the user's explicit permission.
2. In the container: `source scripts/env.sh`, and pass
   `XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS"` to GPU runs.
3. After changes to the pipeline or the model, run the CPU parity check in
   both gradient-reduction modes first:
   `python examples/llama_mpmd_parity.py --platform=cpu --reduce_grads=step`
   and `--reduce_grads=microbatch`.
4. Wrap the run in `timeout -s KILL <seconds>`, write `--log_file` JSONL, and
   send stdout and stderr to `${JAXTITAN_ROOT}/logs/<topic>/<run>.out`.

## Launch

```bash
# Pipeline: 2 stages x 4 GPUs, 1F1B.
XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS" timeout -s KILL 1800 python -m jaxtitan.trainer.train \
  --model=llama3.1-8b --num_stages=2 --schedule=ONE_FWD_ONE_BWD \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=30 \
  --log_file=$LOG.jsonl > $LOG.out 2>&1

# The same step without a pipeline: --parallel=dp or --parallel=fsdp.
```

`--reduce_grads` sets when a stage sums gradients across its GPUs: once per
step (default) or in every microbatch. See `docs/getting_started.md`.

## Monitor

- Compiling a 2-stage pipeline step took 73-114 s on MI355X, since MPMD
  programs bypass the persistent cache; step 0 adds 20-25 s.
- Read the `peak HBM per device` line after step 0.
- Report medians over steps 5 to N.
- Red flags:
  - No step 0 within 3 minutes of `compiled in`: grep the log for
    `may be stuck` with `Initialize clique`. That RCCL hang happens when
    another job uses the same GPUs and does not recover. Stop the run, check
    `rocm-smi --showpids`, and rerun on an idle node.
  - `RESOURCE_EXHAUSTED` in a fragment such as `p33_stage1_fwd_call15`: use
    fewer microbatches or a smaller microbatch size.
  - `Input signature does not match the signature of the compiled function`
    at step 1: see `jaxtitan-debug`.
  - A non-finite loss: the trainer raises `FloatingPointError`; stop and
    compare against the last good run.

## Compare

- Include the no-pipeline baseline (`--parallel=dp` and `--parallel=fsdp`),
  measured on the same day and stack.
- Change one thing at a time. To A/B a code change, run the old variant from
  a `git worktree` with `JAXTITAN_ROOT` exported.
- `--steps` sets the cosine decay length, so compare loss curves only between
  runs with the same `--steps`. GPU runs are not bitwise reproducible; the
  CPU parity check is the numerics reference.
- Put per-step JSONL under `docs/results/` and the table in
  `docs/getting_started.md`.

## After the Job: Free the GPUs and Disk

Do this as soon as a job finishes, fails, hangs or is no longer needed.

```
Cleanup progress:
- [ ] The job is over: the log shows the last step, or you decided to stop it
- [ ] Dry run: scripts/cleanup_after_train.sh [--checkpoint DIR ...]
- [ ] Apply:   scripts/cleanup_after_train.sh --yes [--checkpoint DIR ...]
- [ ] scripts/gpus_free.sh prints free, or lists only other users
- [ ] Large artifacts you created are removed or kept on purpose
- [ ] A SPUR allocation you made is released
- [ ] Report what was stopped and deleted, with sizes, and the GPU state
```

`scripts/cleanup_after_train.sh` touches only the current user's processes
and files. With `--yes` it:
1. stops JAX-Titan processes (trainer, examples, tests): SIGTERM, then
   SIGKILL after 30 s;
2. waits until none of the user's processes hold a GPU;
3. removes the user's stale RCCL segments, `/dev/shm/nccl-*`, which killed
   runs leave behind at about 40 MB each;
4. removes each `--checkpoint DIR` that is below `/dev/shm`, `/tmp` or
   `${JAXTITAN_ROOT}`, owned by the user, and holds an Orbax checkpoint.

Other artifacts to check: profiler traces, IR dumps under
`${JAXTITAN_ROOT}/dumps/`, and A/B worktrees (`git worktree remove`). Keep
JSONL and stdout logs.

Rules:
- Delete only files you created for a test or that the user named. Ask
  before deleting a checkpoint a run may resume from.
- Stopping the host-side `docker exec` client does not stop the process in
  the container; stop the process itself, as the script does.
- On a shared node, list processes before stopping any, and stop only your
  own. Write `pkill -f` patterns as `"[j]axtitan.trainer.train"` so they do
  not match their own command line.
- A stopped process can stay in `rocm-smi --showpids` as a zombie (`Z`) until
  its parent reaps it; it no longer holds GPU memory.

## Rationalizations to Reject

| Excuse | Reality |
|--------|---------|
| "The node looked free a minute ago" | Check again right before the run; another job hangs both |
| "Let it run, step 0 is just slow" | Past compile plus 3 minutes, it is the RCCL hang |
| "Clean up later" | Leftover processes hold 75% of every GPU's memory |
| "That file is probably unused" | Ask, unless you created it for a test |
| "Faster than last week's number" | Compare against a baseline from the same stack and day |

## Additional Resources

- [reference.md](reference.md): run naming, the per-run record and the notes
  format
- `docs/getting_started.md`: container, build, environment, runs and results
- `docs/amd-jax-train-runbook/`: SPUR cluster access and node usage

## Pairing

Pair with `jaxtitan-coding` when code changes are needed, and with
`jaxtitan-debug` when the fault is in jaxlib, XLA, Shardy or ROCm.
