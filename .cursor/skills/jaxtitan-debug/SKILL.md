---
name: jaxtitan-debug
description: >-
  Finds which component owns a JAX-Titan failure (JAX-Titan, jaxlib, Shardy
  MPMD, XLA IFRT IR or PJRT, ROCm and RCCL) and fixes it at the source as an
  upstreamable patch. Use when a stack trace or log points into jaxlib, XLA or
  Shardy, when an MPMD option or schedule does not take effect, when a build
  of the patched wheels is needed, or when deciding where a fix belongs.
---

# JAX-Titan Upstream Debug Guide

## Core Principle

JAX-Titan runs on upstream JAX, Shardy and XLA with a few patches. Find the
component that owns the bug and fix it there, as a patch that can go
upstream. Never work around an upstream bug inside `src/jaxtitan`.

| Owner | Source tree | Fix goes to |
|-------|-------------|-------------|
| JAX-Titan | `src/jaxtitan/` | Normal commit |
| jaxlib (`sdy_mpmd.cc`, `py_mpmd_loaded_executable.cc`) | `${JAXTITAN_ROOT}/src/jax` | `third_party/patches/jaxlib-*.patch` |
| Shardy MPMD: Python API, partitioner, jaxlib glue (`mpmd_program.cc`) | `${JAXTITAN_ROOT}/src/shardy` | `third_party/patches/shardy-*.patch` |
| XLA: IFRT IR (`xla/python/ifrt/ir`), PJRT (`xla/pjrt`), GPU compiler, ROCm profiler | `${JAXTITAN_ROOT}/src/xla` | `third_party/patches/xla-*.patch` (none today) |
| MaxDiffusion model code (MaxText needs no patch today) | `${JAXTITAN_ROOT}/src/maxdiffusion` | `third_party/patches/maxdiffusion-*.patch` |
| TE and JAX-AITER kernels | the TE wheel; `third_party/jax-aiter` | Report upstream; meanwhile make the provider's `supports()` decline the case |
| ROCm, RCCL, driver | none | File an issue; work around only in the environment |

## Finding the Owner

1. Frames in `jaxtitan/` only: JAX-Titan.
2. `mpmd_program.cc`, `lower_to_fragment_calls.cc`, `scheduler.cc`, or the
   `shardy.integrations.python.jax.mpmd` package: Shardy.
3. `program_interpreter.cc`, `IfrtIrProgramCompiler`, `compiled_ifrt_ir_program.cc`:
   XLA IFRT IR.
4. `py_mpmd_loaded_executable.cc`, `sdy_mpmd.cc`: jaxlib, but check the
   arguments JAX-Titan passed first.
5. `rendezvous.cc ... Initialize clique`, `ncclDevKernel`, `hipError`:
   RCCL or ROCm. Rule out another job on the same GPUs first.

A wrong value passed from JAX-Titan into a correct upstream API is a
JAX-Titan bug; fix the caller.

## Known Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `PjRtCompiler requires an HloProgram` | jaxlib without the IFRT-IR-on-PJRT patch | Rebuild with `jaxlib-ifrt-ir-on-pjrt.patch` |
| The ROCm platform is rejected in the IFRT IR atom-program passes | XLA older than openxla/xla#52290 | Build against XLA main at or after `dfca2c9`, as `docs/getting_started.md` documents |
| `ModuleNotFoundError: mlir` from Shardy's `jit.py` | Shardy OSS import | `shardy-mpmd-oss-mlir-import.patch` |
| A partitioning option has no effect | jaxlib's `ParsePartitioningOptions` dropped it | `shardy-mpmd-parse-partitioning-options.patch`; check Shardy's log of the per-mesh fragment order |
| Microbatch 0's backward runs last in 1F1B | Optimizer update merged before scheduling | `mpmd_merge_inferred_after_scheduling` (default in `make_config`) |
| `No module named 'torch'` importing MaxDiffusion's FLUX.1 | MaxDiffusion imports its PyTorch weight converter at module level | `maxdiffusion-optional-torch.patch` |
| FLUX.1-dev fails with a broadcast error in `time_text_embed` or its first block | MaxDiffusion embeds timestep and guidance twice since `64450df4` | `maxdiffusion-flux1-guidance-embedding.patch` |
| `Batching rule for 'custom_partitioning' not implemented` | A JAX-AITER op under `jax.vmap`, as stages run with `--reduce_grads=step` | `--per_device=shard_map`; the provider declines under vmap |
| Segfault (exit 139) in JAX-AITER flash attention's backward | ROCm TE loaded in the same process; both bundle AITER's FMHA | Pick one of them for attention; the provider declines when TE is loaded |
| `is not a axis of the given mesh` from TE fused attention | TE's default `context_parallel_axis=""` with sharded q, k, v | Pass `context_parallel_axis=None`, as `ops/providers/te.py` does |
| `Global mesh resource is not set` from a TE norm or GEMM | TE partitioners need a `MeshResource` | `ops/providers/te.py` sets an empty one on import |
| A provider op passes alone but is wrong under vmap or sharded | Kernel batching or partitioning rule (TE and AITER RMSNorm under vmap, TE RMSNorm's weight gradient when sharded) | Run `python -m jaxtitan.ops.conformance` in every mode; make `supports()` decline the case |
| JAX-AITER `gemm_fp4_bf16` output uncorrelated with BF16 (relative error 1.4) | Its MXFP4 path at `c290cdd`, on JAX-Titan's stack; its own tests check finiteness only | Do not use `aiter_mxfp4`; report upstream |
| `Input signature does not match the signature of the compiled function` at step 1 | Input shardings differ from what the executable returns, e.g. `P(None, None)` against `P()` | Build specs with `jaxtitan.parallel.spec`, which drops trailing Nones |
| `Received incompatible devices` from a sharding constraint | Constraints in one MPMD program name different meshes | Use one stage mesh for every constraint |
| Step 0 hangs in `Initialize clique` | Another job on the same GPUs | Run on an idle node |
| `Disabling HloShardingV3` warnings | XLA falls back for MPMD fragments | Harmless |
| Cross-mesh copies show as `MemcpyH2D` | rocprofiler labels peer copies that way | No host staging; nothing to fix |

## Fix Workflow

1. Log the bug in `${JAXTITAN_ROOT}/notes/debug-log.md`: symptom, trace,
   owner.
2. Check whether upstream already fixed it:
   `git -C ${JAXTITAN_ROOT}/src/<tree> fetch origin main` and
   `git show origin/main:<path>`.
3. Edit the worktree in `${JAXTITAN_ROOT}/src/<tree>`, never the user's own
   checkouts.
4. Regenerate the patch:
   `git -C ${JAXTITAN_ROOT}/src/<tree> diff -- <paths> > third_party/patches/<name>.patch`,
   and describe it in `third_party/patches/README.md`.
5. Rebuild: `WHEELS=jaxlib scripts/build_jax_rocm.sh` for jaxlib and Shardy
   C++ (2-4 minutes; the script builds against the patched Shardy tree), or
   the full build for XLA changes in the ROCm plugin. Python-only Shardy
   changes need no rebuild.
6. Reinstall:
   `pip install --force-reinstall --no-deps ${JAXTITAN_ROOT}/src/jax/dist/<wheel>`.
7. Verify: the failing case, then `python -m pytest tests/`, then one GPU run
   on an idle node.
8. Record the result in the notes, and list the patch as an upstream PR to
   send (the user sends PRs; never push).

## Hard Rules

- Never monkey-patch upstream modules from `src/jaxtitan`. The only shim is
  `src/jaxtitan/_compat.py`.
- Never copy upstream files into `src/jaxtitan` to work around a bug.
- One patch file per upstreamable change; patches stay Apache-2.0.
- XLA must stay at or after the commit the build documents; the build script
  overrides jax's pinned XLA with `${JAXTITAN_ROOT}/src/xla`.
- Never push, and never target the rocm-xla `main` branch, which is a mirror.

## Pairing

Pair with `jaxtitan-coding` for changes in `src/jaxtitan`, and with
`jaxtitan-training` for running, measuring and cleaning up jobs.
