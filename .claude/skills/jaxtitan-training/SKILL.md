---
name: jaxtitan-training
description: >-
  Runs, monitors, compares and cleans up JAX-Titan training jobs on AMD
  Instinct GPUs. Use when launching or benchmarking jaxtitan.trainer.train or
  the examples, when a run hangs, slows down, runs out of memory or diverges,
  when comparing pipeline schedules or layouts against the no-pipeline
  baseline, and after a job finishes, fails or is stopped, to stop its
  processes, free the GPUs and remove checkpoints.
---

# JAX-Titan Training Guide

## Core Principle

GPU time on a shared node is the scarce resource. Start a run only on an idle
node, put a timeout on every run, judge every result against a baseline
measured on the same stack, and free the GPUs and disk as soon as a job is
over. Keep notes in `${JAXTITAN_ROOT}/notes/debug-log.md` (`JAXTITAN_ROOT`
defaults to `.jaxtitan` next to the repo); read it at the start of every new
debug session.

## Before a Run

1. On the host, `scripts/gpus_free.sh` must print `free`. If it prints
   `busy`, wait or ask the user. Never stop another user's process without
   the user's explicit permission.
2. In the container: `source scripts/env.sh`, and pass
   `XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS"` to GPU runs.
3. After code changes, run the CPU tests first: `python -m pytest tests/`.
4. Plan checkpoint space: a Llama 3.1 8B checkpoint is 80 GB and the trainer
   keeps the latest two. `/dev/shm` is host RAM, so do not leave checkpoints
   there.
5. Wrap the run in `timeout -s KILL <seconds>`, record it with `--run_dir`
   (step records and timeline events, `env.json`, GPU power, `results.json`;
   see `docs/metrics.md`), and send stdout and stderr to
   `${JAXTITAN_ROOT}/logs/<topic>/<run>.out`.
6. To answer how long a training run takes on given GPUs, or whether it fits,
   use `python -m jaxtitan.planning.budget` (CPU only; `docs/planning.md`).
   Pass `--from_run` with a measured run directory rather than relying on the
   assumed 40% MFU, and say which MFU the estimate used.

## Launch

```bash
# Pipeline: 2 stages x 4 GPUs, zero-bubble schedule (default).
export JAXTITAN_LAUNCH_TIME=$(date +%s.%N)
XLA_FLAGS="$JAXTITAN_ROCM_XLA_FLAGS" timeout -s KILL 1800 python -m jaxtitan.trainer.train \
  --model=llama3.1-8b --num_stages=2 --microbatches=8 --microbatch_size=4 \
  --seq_len=2048 --steps=30 --run_dir=$RUN > $RUN.out 2>&1

# The same step without a pipeline: --parallel=dp or --parallel=fsdp
# (adapter models: --parallel=dp).

# Models from MaxText and MaxDiffusion (FLUX.1 trains on synthetic batches).
... --model=maxtext:llama3.1-8b --num_stages=2 ...
... --model=maxdiffusion:flux-dev --num_stages=4 --steps=20 --lr=1e-5 ...
```

Layout flags: `--schedule`, `--stage_fsdp`, `--tensor_parallel`,
`--stage_layers`, `--stages_per_mesh`, `--reduce_grads`. Checkpoints:
`--checkpoint_dir`, `--checkpoint_every`; rerunning resumes. See
`docs/pipelines.md`. Operators: `--ops=attention=te` (TE fused attention) or
`--ops=attention=aiter`; TE's FP8 GEMMs and every JAX-AITER op need
`--per_device=shard_map` with per-step reduction, and the `ops:` lines after
`compiled in` show what each op ran on. `maxtext:` and `maxdiffusion:` models support only data
parallelism inside stages with gradients reduced per microbatch (the
default for them); `--stage_layers` counts their decoder layers or FLUX.1
blocks, and `--model_overrides` takes JSON config overrides.

## Monitor

- Compiling a 2-stage pipeline step takes 66-116 s, since MPMD programs
  bypass the persistent cache; step 0 adds 20-25 s.
- Read the `peak HBM per device` line after step 0.
- Report step time from `python -m jaxtitan.observability.summarize RUN`
  (p50, p95 and max over steady steps; `--skip` drops warmup). Exclude
  profiled steps (`--profile_steps`): the first traced step carries profiler
  startup cost.
- Red flags:
  - No step 0 within 3 minutes of `compiled in`: grep the log for
    `may be stuck` with `Initialize clique`. That RCCL hang happens when
    another job uses the same GPUs and does not recover. Stop the run, check
    `rocm-smi --showpids`, and rerun on an idle node.
  - `RESOURCE_EXHAUSTED` in a fragment such as `p33_stage1_fwd_call15`: use
    fewer microbatches, `--stage_fsdp`, or `--tensor_parallel`. FLUX.1-dev
    needs 4 stages: on 2, the first stage's 7.9B parameters with fp32 AdamW
    state reach 272 GiB per GPU.
  - `Input signature does not match the signature of the compiled function`
    at step 1: see `jaxtitan-debug`.
  - A non-finite loss or gradient norm: the step made no update, nothing is
    saved after it, and the trainer raises `FloatingPointError`; stop and
    compare against the last good run.
  - `step watchdog: ... exiting` and exit code 124: a step ran past
    `--step_timeout` (or 10x the median); the log holds every thread's stack.
    See `jaxtitan-debug`.

## Compare

- Include the no-pipeline baseline (`--parallel=dp` and `--parallel=fsdp`),
  measured on the same day and stack. Check every new pipeline layout with
  Gate 0 against it: `summarize --gate0 REFERENCE RUN --gate0_steps 10
  --gate0_rtol 2e-3`.
- For a published number, run three repeats and report the median and
  spread (`summarize` prints both and a regression threshold), and MFU
  against the calibrated ceiling too (`--ceilings`, from
  `python -m jaxtitan.observability.ceilings`).
- Change one thing at a time. To A/B a code change, run the old variant from
  a `git worktree` with `JAXTITAN_ROOT` exported.
- `--steps` sets the cosine decay length, so compare loss curves only between
  runs with the same `--steps`. GPU runs are not bitwise reproducible; the
  CPU parity check (`examples/llama_mpmd_parity.py`) is the numerics
  reference.
- Put per-step JSONL under `docs/results/<date>/` and the table in
  `docs/pipelines.md`.

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
   `${JAXTITAN_ROOT}`, owned by the user, and holds an Orbax checkpoint;
5. removes the user's GPU core dumps, `gpucore.<pid>.gpu`, from the
   repository and `${JAXTITAN_ROOT}`.

Other artifacts to check: profiler traces under `--profile_dir`, IR dumps
under `${JAXTITAN_ROOT}/dumps/`, and A/B worktrees (`git worktree remove`).
Keep JSONL and stdout logs.

Rules:
- Delete only checkpoints you created for a test or that the user named.
  Ask before deleting one a run may resume from.
- Stopping the host-side `docker exec` client does not stop the process in
  the container; stop the process itself, as the script does.
- On a shared node, list processes before stopping any, and stop only your
  own. Write `pkill -f` patterns as `"[j]axtitan.trainer.train"` so they do
  not match their own command line.
- A stopped process can stay in `rocm-smi --showpids` as a zombie (`Z`) until
  its parent reaps it; it no longer holds GPU memory.
- After a GPU memory fault (`Memory access fault by GPU node-N`), ROCm writes
  all GPU memory to `gpucore.<pid>.gpu` in the process's working directory:
  502 GB from one faulting kernel on 8x MI355X, enough to fill the shared
  disk. `scripts/env.sh` sets `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` to prevent
  it. If a dump appears anyway, delete it at once (`rm gpucore.<pid>.gpu`, or
  the cleanup script) and check `df -h /`; do not wait for the job to end.

## Rationalizations to Reject

| Excuse | Reality |
|--------|---------|
| "The node looked free a minute ago" | Check again right before the run; another job hangs both |
| "Let it run, step 0 is just slow" | Past compile plus 3 minutes, it is the RCCL hang |
| "Clean up later" | Leftover processes hold 75% of every GPU's memory |
| "That checkpoint is probably unused" | Ask, unless you created it for a test |
| "Keep the GPU core dump to debug the fault" | It holds all GPU memory, hundreds of GB on a shared disk; reproduce the fault on one GPU with small inputs |
| "Faster than last week's number" | Compare against a baseline from the same stack and day |

## Additional Resources

- [reference.md](reference.md): run naming, the per-run checklist and the
  notes format
- `docs/pipelines.md`: schedules, layouts, checkpoints and measured results
- `docs/planning.md`: training budgets (time, GPU-hours, energy, memory) per
  GPU type
- `docs/getting_started.md`: container, build and environment
- `docs/amd-jax-train-runbook/`: SPUR cluster access and node usage

## Pairing

Pair with `jaxtitan-coding` when code changes are needed, and with
`jaxtitan-debug` when the fault is in jaxlib, XLA, Shardy or ROCm.
