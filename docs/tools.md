# Tools

JAX-Titan's scripts, command-line tools and agent skills, grouped by the
stage of a training job they serve. Each entry says what the tool answers and
links to the page that documents it.

## A job, stage by stage

1. **Set up once.** Start the container with `scripts/container.sh` on the
   host, build the wheels, and `source scripts/env.sh` in every shell in the
   container ([getting_started.md](getting_started.md)).
2. **Check the node.** On the host, `scripts/preflight_dev.sh` shows disk
   space and who holds each GPU, and `scripts/gpus_free.sh` must print
   `free`. After a code change, run `python -m pytest tests/` first.
3. **Plan.** `planning.budget`, `planning.step` and `observability.flops`
   estimate time, memory and the stage split, on CPU.
4. **Run.** `python -m jax_titan.trainer.train` under `timeout`, with
   `--run_dir` so that the run is recorded
   ([metrics.md](metrics.md#recording-a-run)).
5. **Clean up.** `scripts/cleanup_after_train.sh` shows what it would stop and
   remove; `--yes` applies it.
6. **Read the run.** `observability.summarize`, `profile`, `plot` and `ettr`.
   A measured run makes the next plan more accurate through
   `planning.budget --from_run` and `planning.step --calibrate`.

## Scripts

Scripts marked host run outside the container, where they see every user's
processes. The others need the container's TheRock libraries.

| Script | Runs on | What it does |
|---|---|---|
| `container.sh` | host | Starts the TheRock 10.1 development container with all GPUs and a user with your host uid, so files in the workspace keep your owner ([getting_started.md](getting_started.md#1-container)) |
| `env.sh` | container | Activates the venv; sets TheRock's library path, `PYTHONPATH` and the JAX compilation cache; turns off GPU core dumps; sets `JAX_TITAN_ROCM_XLA_FLAGS` for GPU runs. Source it in every shell |
| `build_jax_rocm.sh` | container | Builds the jax, jaxlib and ROCm plugin wheels against `${JAX_TITAN_ROOT}/src/xla` and the patched Shardy; `WHEELS=jaxlib` rebuilds only jaxlib ([getting_started.md](getting_started.md#3-build)) |
| `build_jax_aiter.sh` | container | Builds JAX-AITER from its submodule against the installed jaxlib, with prebuilt AITER JIT libraries |
| `build_rccl_inspector.sh` | container | Ports RCCL's Inspector profiler plugin to HIP, for the trainer's `--rccl_inspector` ([metrics.md](metrics.md#recording-a-run)) |
| `gpus_free.sh` | host | Prints `free` and exits 0 when no process holds a GPU and every GPU is idle; prints what is busy otherwise |
| `preflight_dev.sh` | host | Reads only: free space on each filesystem, each GPU's load and the users holding it, the top users by disk and the largest containers. Exits 1 when a process holds a GPU, a GPU is busy, or a filesystem is below 10% free. `--top N` sets the list length; `--no-docker` skips container sizes |
| `cleanup_after_train.sh` | host or container | Prints what it would do: stop your JAX-Titan processes, wait until they release the GPUs, and remove your stale `/dev/shm/nccl-*` segments, the Orbax checkpoints named with `--checkpoint`, and your GPU core dumps. `--yes` applies it |
| `bench_collectives.sh` | container | Measures RCCL (and MORI-EP when installed) time and bus bandwidth by message size on this node, for the planner. Needs free GPUs ([planning.md](planning.md#collectives-rccl-and-mori)) |

`preflight_dev.sh` and `cleanup_after_train.sh` read `USER`, which
`docker exec` does not set; export it first in a shell started that way. The
cleanup's process pattern also matches your other `pytest` and `tests/`
processes, so read the dry run's list before `--yes`.

## Command-line tools

Run each with `python -m jax_titan.<module>`, or `python <path>` for the
examples. The planners, and the tools that read a finished run (`summarize`,
`profile`, `plot`, `ettr`, `collectives`, `perfdb`), need no GPU.

### Correctness

| Command | What it checks |
|---|---|
| `python -m pytest tests/` | On CPU with 8 devices: schedules and layouts against `jax.jit`, exact checkpoint resume, the MaxText and MaxDiffusion adapters (skipped without their extras), the planners, and the agent skills' two copies |
| `python examples/<name>_parity.py --platform=cpu` | A pipeline step against one `jax.jit` program, and the adapter stages against MaxText's and MaxDiffusion's own models ([getting_started.md](getting_started.md#5-runs)) |
| `ops.conformance` | Operator providers against native JAX: outputs and gradients on one GPU, under `jax.vmap`, sharded, inside `shard_map` and inside an MPMD step, and their timings ([getting_started.md](getting_started.md#5-runs)) |

### Measuring a run

[metrics.md](metrics.md#tools) has their options and the metrics they report.

| Command | What it answers |
|---|---|
| `observability.summarize RUN` | Step-time percentiles, MFU and HFU, startup and checkpoint costs; across runs, repeats and scaling. `--gate0 REFERENCE RUN` compares losses against a reference run |
| `observability.profile TRACE_DIR` | Each GPU's compute, exposed collective, copy and idle shares from a `jax.profiler` trace |
| `observability.plot RUN` | Figures of MFU, step time, loss, gradient norm and power over the run |
| `observability.ettr RUN` | ETTR, and the checkpoint interval and ETTR projected to more nodes |
| `observability.collectives DIR` | RCCL Inspector completion times by operation and message size |
| `observability.power OUT.jsonl` | GPU power, energy, clocks and throttling, sampled beside any job |
| `observability.env OUT.json` | What a run ran on: packages, commits, flags, GPUs and the rank map |
| `observability.ceilings OUT.json` | The GEMM and HBM rates XLA reaches on one GPU, for MFU against an attainable peak (needs a GPU) |
| `observability.detectors OUT -- ARGS` | The step-time cost of each failure detector; runs the trainer once per detector |
| `observability.perfdb perf.jsonl` | Runs slower than the best earlier run with the same model, GPU and layout |

### Planning

| Command | What it answers |
|---|---|
| `planning.budget` | Time, GPU-hours, energy, cost and memory of a run on one or more GPU types; `--from_run` uses a measured run instead of an assumed MFU ([planning.md](planning.md)) |
| `planning.step` | Step time by part: compute, collectives, bubble and runtime gaps. `--calibrate RUN` fits the parts to a run; `--ladder` predicts other GPU counts, with `--weak` keeping the batch per GPU ([planning.md](planning.md#step-time-calibration-and-the-scale-ladder)) |
| `planning.collectives` | RCCL and MORI time by message size; `--calibrate` turns `bench_collectives.sh` output into a table ([planning.md](planning.md#collectives-rccl-and-mori)) |
| `planning.remaining RUN` | Time left in a running job ([planning.md](planning.md#during-a-run-time-left-and-the-performance-database)) |
| `observability.flops` | Model and executed FLOPs per op and per stage, and the layer split that balances the stages ([metrics.md](metrics.md#tools)) |

## Agent skills

Three skills hold the project's rules for coding agents, with identical
content in `.claude/skills/` (Claude Code) and `.cursor/skills/` (Cursor).
An agent working in this repository loads a skill when the task matches its
description; you can also name it in the request, for example "use
jax_titan_training to run the no-pipeline baseline".

| Skill | Use it when | What it holds |
|---|---|---|
| `jax_titan_coding` | Writing or reviewing code in this repository | Architecture rules (stages, meshes, shardings, operators, adapters, precision), testing rules (a new schedule, layout or model matches `jax.jit`), and code style |
| `jax_titan_training` | Launching, monitoring, comparing or cleaning up a training job | Checks before a run, launch commands, red flags while it runs, how to compare against a baseline, and a cleanup checklist; `reference.md` adds run names, what to record for each result and the notes file |
| `jax_titan_debug` | A stack trace or log points into jaxlib, XLA, Shardy or ROCm | Which component owns a failure, known failures and their fixes, and the workflow for an upstream patch |

Edit both copies of a skill: `tests/test_agent_skills.py` fails when they
differ. The SPUR runbook adds rules for agents on that cluster
([amd-jax-train-runbook/README.md](amd-jax-train-runbook/README.md#for-agents)).
