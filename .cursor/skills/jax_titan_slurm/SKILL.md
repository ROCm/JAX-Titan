---
name: jax_titan_slurm
description: >-
  Submits, monitors and cleans up one-node JAX-Titan training jobs on a Slurm
  cluster with sbatch, from the login node. Use when asked to run, test or
  benchmark JAX-Titan (a Llama, MaxText or FLUX.1 model) through Slurm,
  sbatch or a cluster scheduler; when a submitted job stays pending and you
  need to know why or how long; when the job's container, wheel build, venv
  or trainer fails on the node; when setting up the source trees for a first
  job; and when adapting the scripts to a new cluster. Covers
  scripts/setup_sources.sh, scripts/slurm_submit.sh and scripts/slurm_job.sh.
---

# JAX-Titan with Slurm sbatch

## Core Principle

A submitted job holds 8 GPUs that other people are queueing for. Submit only
when the user asked for a run, submit one job per run (cancel the others as
soon as one starts), give every job a time limit, watch it until it ends,
and confirm that it ended `COMPLETED 0:0` with its container removed. Never
`scancel` another user's job, and never ssh to compute nodes to work around
the scheduler.

## Requirements

The scripts use only standard sbatch, squeue, sacct, scontrol and scancel.
What limits where they run is the cluster, not Slurm:

| Requirement | Why | Checked by |
|---|---|---|
| docker usable by the job user on the node | The job starts the TheRock container with `docker run` | `slurm_job.sh`, exit 7 |
| AMD GPUs (`/dev/kfd`, `rocm-smi`), 8 per node | ROCm container and `scripts/gpus_free.sh` | `slurm_job.sh`, exit 7 |
| Repository and `JAX_TITAN_ROOT` on a filesystem shared by the login and compute nodes, at the same path | Wheels, venv and logs are built once and reused; the container mounts the repository's parent | `slurm_job.sh`, exit 7 |
| Internet on the node for the first job | Image pull, Bazel downloads, pip | the build log |
| A host ROCm driver new enough for TheRock 10.1 | The image brings its own ROCm user space | the `gpu check` stage, exit 6 |

A cluster without docker (Apptainer, enroot) or without internet needs a
different runtime or prebuilt wheels: say so, do not work around it.

## Cluster settings

Partition, QOS, account, scratch path and how the cluster asks for GPUs
differ per cluster. Keep them in a file outside the repository, named by
`JT_SITE_ENV`, which `slurm_submit.sh` and the job source:

```bash
# ~/jax_titan_site.env
JT_PARTITION=<partition>
JT_QOS=<qos>                      # optional
JT_ACCOUNT=<account>              # optional
JT_SCRATCH=/local/$(id -un)/jax_titan   # node-local; default $SLURM_TMPDIR or /tmp
JT_GPU_REQUEST=--gres=gpu:8       # default --gpus-per-node=8
JT_SIGNAL=                        # empty if sbatch has no --signal
```

If the user keeps such a file, use it; ask for its path before guessing
cluster values. Command-line options (`-p`, `-q`, `-A`, `-t`) override it.

## How the pieces fit

| Step | Where | Command | Time |
|---|---|---|---|
| Source trees, once | login node | `scripts/setup_sources.sh` | 1-3 min |
| Submit | login node | `scripts/slurm_submit.sh [-p P] [-q Q] [-A A] [-t T] -- TRAINER FLAGS` | seconds |
| Job | compute node | `scripts/slurm_job.sh` (sbatch runs it) | see below |

`slurm_job.sh` prints each stage as `=== <time> <stage>` and skips a setup
stage whose output exists on the shared filesystem:

1. `node`: host, scratch, the requirements above, the source trees,
   `gpus_free.sh` (exit 3 when another process holds a GPU).
2. `container`: pulls the image (about 1.5 min), starts
   `<user>_therock-10.1_jax_titan_<JobID>` with the job user's uid.
3. `build wheels`: only when `${JAX_TITAN_ROOT}/src/jax/dist` lacks them;
   about 10 min on 236 CPUs, log in `logs/build_jax_rocm.log`.
4. `venv`: only when `${JAX_TITAN_ROOT}/venv/.jax_titan_ok` is missing;
   about 2 min.
5. `gpu check`: the jax version and 8 `RocmDevice`s.
6. `train`: the trainer under `timeout -s KILL $TRAIN_TIMEOUT` (default
   45m), step records in `logs/slurm/<JobID>.jsonl`. Skipped with
   `--setup-only`.
7. `cleanup`: by an EXIT trap, also after SIGTERM (scancel, or the time
   limit with `JT_SIGNAL`): `cleanup_after_train.sh --yes` in the
   container, removes the container, waits up to 60 s for `gpus_free.sh`,
   prints `GPU state after N s: free` and `job exit=<rc>`.

Exit codes: 2 missing source trees, 3 GPUs busy, 4 build failed, 5 venv
failed, 6 no GPUs in JAX, 7 a requirement is missing, 143 SIGTERM; otherwise
the trainer's (137 = `TRAIN_TIMEOUT`).

## Before Submitting

1. `scripts/setup_sources.sh` once: jax and xla become detached worktrees of
   sibling `../jax` and `../xla` checkouts when they exist (the user's
   checkouts do not change), Shardy is cloned at the commit XLA pins, and the
   patches are applied. Existing trees are left alone.
2. `df -h` the shared filesystem: sources, wheels and venv take about 3 GB,
   an 8B checkpoint 80 GB.
3. Pick partition, QOS and account, and read why a job pends:

```bash
sinfo -t idle
sacctmgr show qos                                 # GrpTRES node=N per QOS
scontrol show job <JobID> | grep -E 'Account|QOS|JobState|Reason'
squeue -A <account> -t R -h -o '%i %L %D %u'      # the group's running jobs, time left
scontrol show partition <P> | grep AllowAccounts
squeue -w <node>                                  # what really runs on an "idle" node
```

- `QOSGrpNodeLimit`: the QOS's group limit is reached; the job waits for the
  group, however many nodes are idle. Another QOS the account may use (often
  a low-priority, preemptible one) can start sooner.
- `Resources`: no node is free; jobs ahead at the same priority go first.
- `AllowAccounts`: do not submit to partitions the user's accounts are not in.

## Submit

```bash
# First job: builds the wheels and the venv, then 20 steps of Llama 3.1 8B,
# 2 pipeline stages, synthetic batches.
JT_SITE_ENV=<file> scripts/slurm_submit.sh -t 01:00:00 -- \
  --model=llama3.1-8b --data=synthetic --num_stages=2 --steps=20

JT_SITE_ENV=<file> scripts/slurm_submit.sh --setup-only        # no training
JT_SITE_ENV=<file> scripts/slurm_submit.sh -n -- --steps=20    # print, do not submit
```

Trainer flags after `--` pass through unchanged; see `jax_titan_training` for
models, layouts and operators. `--data=synthetic` needs no dataset or
tokenizer; WikiText needs `python -m jax_titan.data.text` once in the
container and a MaxText checkout next to the repo. `slurm_submit.sh` writes
`JAX_TITAN_ROOT`, `JT_SITE_ENV`, `JT_IMAGE`, `JT_SCRATCH`, `TRAIN_TIMEOUT` and
`BAZEL_CPUS` into the job's command, so the scheduler need not export them.

## Monitor

Logs: `${JAX_TITAN_ROOT}/logs/slurm/<JobID>.out` and `.err`. ROCm prints
`queue-intercept` warnings and Shardy prints `I` lines naming `train_step`
fragments; filter them:

```bash
tail -F $LOGS/<JobID>.out $LOGS/<JobID>.err | grep -v -E 'queue-intercept|^I[0-9]{4} |^\s+- ' \
  | grep -E --line-buffered '^=== |job exit=|^step |compiled in|Traceback|error|RESOURCE_EXHAUSTED|Killed|may be stuck'
```

Expected for Llama 3.1 8B, 2 stages, 8 microbatches of 4 x 2048 on 8x
MI355X: `compiled in` about 113 s, step 0 about 25 s, then 1.56-1.70 s per
step (39-42k tok/s, MFU about 9.5%); loss starts near 12.59 on synthetic
batches and falls slowly; the gradient norm can jump to about 100 after
warmup and come back. The red flags of `jax_titan_training` (no step 0
within 3 minutes of `compiled in`, `RESOURCE_EXHAUSTED`, a non-finite loss)
apply.

When it ends:

```bash
sacct -j <JobID> -o JobID,State,ExitCode,Elapsed,NodeList   # COMPLETED 0:0
grep -E 'GPU state after|job exit=' $LOGS/<JobID>.out       # free, exit=0
```

## Failures

| Symptom | Cause and fix |
|---|---|
| Exit 2, `missing .../src/<tree>` | Run `scripts/setup_sources.sh` on the login node |
| Exit 3, `GPUs are busy` | Another process holds a GPU on that node; resubmit with `SBATCH_EXTRA="-x <node>"` |
| Exit 7, `docker is not usable` | The cluster does not give jobs docker: stop and tell the user |
| sbatch rejects `--signal` or `--gpus-per-node` | Set `JT_SIGNAL=` or `JT_GPU_REQUEST=--gres=gpu:8` in the cluster settings |
| Exit 4, build failed | Read the end of `logs/build_jax_rocm.log`; a patch that no longer applies means the trees moved off the tested revisions |
| Download 404 from `mirror.bazel.build` | Normal: Bazel falls back to the next URL |
| `CANCELLED` under a preemptible QOS | Preempted; resubmit, the finished setup stages are skipped |
| Exit 137 | `TRAIN_TIMEOUT` killed the trainer; raise it or lower `--steps` |
| Killed at the time limit without `job exit=` | SIGKILL came before cleanup: a container may be left holding the GPUs; set `JT_SIGNAL` where sbatch supports it, or a `-t` with headroom |

## Rationalizations to Reject

| Excuse | Reality |
|--------|---------|
| "sinfo says idle, so it starts now" | Check `squeue -w <node>`; some schedulers show a busy node as idle |
| "Keep all the submissions in case one is slow" | Each running one holds 8 GPUs; cancel the rest as soon as one runs |
| "The job printed the last step, it is done" | Confirm `job exit=0`, `COMPLETED 0:0` and the GPU state after cleanup |
| "Put the cluster's partition in the script" | Cluster values go in `JT_SITE_ENV`, outside the repository |

## Pairing

Pair with `jax_titan_training` for trainer flags, red flags while a run goes
and comparing results, and with `jax_titan_debug` when a failure points into
jaxlib, XLA, Shardy or ROCm.
