---
name: jaxtitan-coding
description: >-
  Architecture rules and coding standards for JAX-Titan, the JAX-native MPMD
  training framework (Shardy MPMD pipelines run as IFRT IR programs on ROCm).
  Use when writing, reviewing or changing code in the JAX-Titan repo: pipeline
  stages and schedules, shardings, models, the trainer, examples, tests and
  docs.
---

# JAX-Titan Coding Guide

## Core Principle

JAX-Titan is JAX-native: it builds on `jax.jit`, Shardy MPMD, IFRT IR, PJRT
and XLA, with no PyTorch runtime. A pipeline is one Shardy MPMD program, not
hand-written point-to-point code. Changes to jaxlib, XLA or Shardy are
upstreamable patches in `third_party/patches/`, never workarounds in
`src/jaxtitan`. `src/jaxpp/` is the earlier JaxPP project, kept unchanged for a
later comparison: do not import from it or edit it.

## Architecture Rules

- **Stages and microbatches:** each stage is an `mpmd.named_computation`
  assigned to a mesh through `jaxtitan.mpmd.make_config`; each microbatch is
  an `mpmd.call` with its own call counter (`jaxtitan.mpmd.pipeline_loss`).
  Schedules come from Shardy's pipeline registry.
- **Meshes:** `make_topology` gives every stage its own mesh with a `data`
  axis. Sharding constraints inside one MPMD program must all use one stage
  mesh; Shardy MPMD only reads their axes.
- **Shardings:** a train step's input shardings must equal its out_shardings
  exactly, for example `P()` for replicated state, or the executable's call
  cache misses at step 1. Stage state is replicated on its mesh; the
  no-pipeline FSDP baseline shards the first evenly splitting dimension that
  is not the layer axis (`_fsdp_spec`).
- **Gradient reduction:** with `--reduce_grads=step`, each data-parallel rank
  trains its own parameter copy and stage functions must not add sharding
  constraints. With `microbatch`, stage functions constrain activations at
  stage boundaries.
- **Operators:** models call semantic ops from `jaxtitan.ops` (`attention`,
  `rms_norm`, `swiglu`), never a kernel library directly.
- **Precision:** parameters and optimizer state are fp32; compute casts to
  bf16 before the layer scan. Casting inside the rematerialized layer
  instead was measured 2-3% slower with more memory per GPU; do not move it
  without a new A/B.
- **Entry points** import jax inside `main()`, after `--platform=cpu` sets
  `JAX_PLATFORMS`.

## Testing Rules

- `examples/llama_mpmd_parity.py --platform=cpu` must pass in both
  gradient-reduction modes before every commit that touches the pipeline or
  a model: the pipeline step must match single-mesh `jax.jit` in fp32 with
  SGD.
- A new schedule or model gets a parity run with its flags.
- `python -m pytest tests/` must pass.
- A performance claim needs a GPU run on an idle node against the
  no-pipeline baseline from the same stack (see `jaxtitan-training`), with
  its JSONL in `docs/results/`.

## Code Style

- Match the surrounding code: 2-space indentation, prose docstrings that say
  what a function returns and what its caller must know, type hints on
  public functions, `# pylint: disable=g-import-not-at-top` on deferred
  imports.
- Comments only state constraints the code cannot show, never what the next
  line does or why a change is correct.
- Spell the project `JAX-Titan`. Write docs in plain technical English with
  complete sentences, and keep measured numbers in `docs/getting_started.md`.
- Commit locally in logical pieces with a message that says what changed and
  why. Never push; the user pushes. Never edit the user's own `jax` or `xla`
  checkouts; use the worktrees in `${JAXTITAN_ROOT}/src`.

## Keeping Skills in Sync

The skills exist twice, in `.cursor/skills/` and `.claude/skills/`, with
identical content. Edit both; `tests/test_agent_skills.py` fails when they
differ.

## Pairing

Pair with `jaxtitan-training` to run and measure a change, and with
`jaxtitan-debug` when the fault is in jaxlib, XLA, Shardy or ROCm.
