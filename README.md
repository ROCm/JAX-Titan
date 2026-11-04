# JAX-Titan

JAX-Titan is a JAX-native MPMD training framework, built AMD-native first.

Each pipeline stage is a [Shardy MPMD](https://github.com/openxla/shardy/blob/main/rfcs/2025-06-18-mpmd-rfc.md)
named computation assigned to its own device mesh. The whole train step
compiles to one IFRT IR program: one XLA executable per stage fragment, plus
the cross-mesh transfers between them. A single controller runs it with XLA's
IFRT IR interpreter on the ROCm PJRT plugin, without Pathways.

> **Status:** early development. APIs will change.

## What works

- Single-controller Shardy MPMD execution on ROCm, with no Pathways service.
- Pipeline parallelism with Shardy's schedules (`ONE_FWD_ONE_BWD`, `GPIPE`)
  and microbatching, data parallelism inside each stage, and per-stage AdamW
  state that stays on the stage's mesh.
- Llama 3 models using MaxText's `llama3.1-8b` configuration, WikiText-103
  with the Llama 3 tokenizer, and throughput and MFU reporting.
- Checks against single-mesh `jax.jit`: the MPMD pipeline step matches it in
  loss and parameters.

First result on 8x AMD Instinct MI355X: Llama 3.1 8B with 2 pipeline stages of
4 GPUs, the 1F1B schedule and 65,536 tokens per step trained at 2.20 s per step
(29.8k tokens/s, 7.0% MFU). Loss fell from 12.6 to 6.35 in 100 steps. See
[docs/getting_started.md](docs/getting_started.md).

```python
from jaxtitan import mpmd as jt_mpmd
from shardy.integrations.python.jax import mpmd

topology = jt_mpmd.make_topology(num_stages=2)            # mesh0, mesh1
config = jt_mpmd.make_config(topology, "ONE_FWD_ONE_BWD")
train_step = jt_mpmd.make_train_step(stage_fns, optimizer)
compiled = mpmd.jit(
    train_step, config, in_shardings=..., out_shardings=...
).lower(params, opt_states, tokens, targets).compile()
```

## Requirements

JAX-Titan needs three small patches on top of upstream, kept in
[third_party/patches](third_party/patches):

- **jaxlib:** compile IFRT IR MPMD programs on PJRT-backed IFRT clients.
  Upstream jaxlib only compiles them through Pathways.
- **XLA:** accept the ROCm platform in the IFRT IR atom-program passes.
- **Shardy MPMD JAX API:** an import fix for open-source builds.

[docs/getting_started.md](docs/getting_started.md) covers the TheRock
container, the from-source JAX build and the Python environment.

## Layout

| Path | Contents |
|---|---|
| `src/jaxtitan/mpmd` | Topology, Shardy MPMD config and pipelined train steps |
| `src/jaxtitan/models` | Llama 3, with parameters grouped by pipeline stage |
| `src/jaxtitan/ops` | Semantic operators (native JAX today) |
| `src/jaxtitan/data` | Llama 3 tokenizer and pretokenized text sampling |
| `src/jaxtitan/trainer` | Training entry point: `python -m jaxtitan.trainer.train` |
| `examples/` | MPMD spike and parity checks |
| `scripts/` | Container, build, environment and GPU-availability scripts |
| `third_party/patches/` | jaxlib, XLA and Shardy patches |
| `src/jaxpp/` | JaxPP project, kept as-is for a later comparison |

## Roadmap

- Operator providers: ROCm Transformer Engine, JAX-AITER, jax-triton, MORI-EP
  for MoE dispatch/combine, and FlyDSL later.
- Checkpointing with Orbax, FSDP and TP inside stages, and the advanced
  pipeline schedules Shardy provides (zero-bubble, circular).
- Multi-node execution, MaxText and MaxDiffusion model adapters, and RL
  post-training with Ray and vLLM.

## License

MIT, see [LICENSE](LICENSE). Two directories keep the Apache License 2.0:

- `src/jaxpp/`: the original JaxPP project, with its own `LICENSE` and file
  headers.
- `third_party/patches/`: patches to jax, XLA and Shardy, which are Apache-2.0
  projects. See [third_party/patches/README.md](third_party/patches/README.md).
