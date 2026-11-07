"""Checks operator providers against native JAX, and times them.

For each op, provider and case, the provider and `native` run in bf16 on the
same inputs, and both are compared with `native` in float32: the output, and
the gradient of every input for a random cotangent. The provider passes when
its error is at most twice native bf16's, or below its quantization tolerance
(MXFP8, FP8 and MXFP4 GEMMs). Modes:

  single   one device, under jax.jit
  vmap     under jax.vmap over a leading axis, as stages run when gradients
           are reduced once per step
  sharded  batch-major inputs sharded over 4 devices
  manual   inside a shard_map over those 4 devices, on local shards, as stages
           run with --per_device=shard_map
  mpmd     inside a 2-stage Shardy MPMD train step on 2 meshes of 4 devices;
           loss and gradients against the same step on native
  bench    forward plus backward time on one device, against native

  python -m jaxtitan.ops.conformance --platform=rocm --providers=te,aiter,triton,gluon
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable

import numpy as np

# Relative error allowed per quantized GEMM; ops multiply it by their GEMM count.
TOLERANCE = {"te_mxfp8": 0.06, "te_fp8": 0.06, "aiter_mxfp4": 0.25}
GEMMS = {"swiglu": 2}
MODES = ("single", "vmap", "sharded", "manual", "mpmd", "bench")


@dataclasses.dataclass(frozen=True)
class Case:
  """Inputs for one op: `make(rng, batch)` returns (args, kwargs).

  `batched` indexes the args that lead with the batch axis; the others
  (weights) are shared. `quick` cases are small enough for CPU.
  """

  op: str
  name: str
  make: Callable
  batched: tuple[int, ...] = (0,)
  quick: bool = False


def _normal(rng, shape, scale=1.0):
  return (rng.standard_normal(shape) * scale).astype(np.float32)


def _attention_case(name, s, hq, hkv, d, causal, batch, quick=False):
  def make(rng, b=batch):
    return tuple(_normal(rng, (b, s, h, d)) for h in (hq, hkv, hkv)), {"causal": causal}
  return Case("attention", name, make, (0, 1, 2), quick)


def _norm_case(name, s, e, batch, quick=False):
  def make(rng, b=batch):
    return (_normal(rng, (b, s, e)), 1 + _normal(rng, (e,), 0.1), 1e-5), {}
  return Case("rms_norm", name, make, (0,), quick)


def _swiglu_case(name, s, e, f, batch, quick=False):
  def make(rng, b=batch):
    return (_normal(rng, (b, s, e)), _normal(rng, (e, f), e ** -0.5),
            _normal(rng, (e, f), e ** -0.5), _normal(rng, (f, e), f ** -0.5)), {}
  return Case("swiglu", name, make, (0,), quick)


def _linear_case(name, s, e, n, batch, quick=False):
  def make(rng, b=batch):
    return (_normal(rng, (b, s, e)), _normal(rng, (e, n), e ** -0.5)), {}
  return Case("linear", name, make, (0,), quick)


def _rope_case(name, s, h, d, batch, quick=False):
  def make(rng, b=batch):
    angles = np.arange(s)[:, None] * 10000.0 ** (-np.arange(0, d, 2) / d)
    return (_normal(rng, (b, s, h, d)), np.cos(angles).astype(np.float32),
            np.sin(angles).astype(np.float32)), {}
  return Case("rope", name, make, (0,), quick)


CASES = [
    _attention_case("small", 128, 4, 2, 64, True, 2, quick=True),
    _attention_case("llama3.1-8b", 2048, 32, 8, 128, True, 2),
    _attention_case("flux.1-1024px", 4608, 24, 24, 128, False, 1),
    _norm_case("small", 64, 256, 2, quick=True),
    _norm_case("llama3.1-8b", 2048, 4096, 2),
    _swiglu_case("small", 64, 256, 512, 2, quick=True),
    _swiglu_case("llama3.1-8b", 2048, 4096, 14336, 2),
    _linear_case("small", 64, 256, 256, 2, quick=True),
    _linear_case("llama3.1-8b-wq", 2048, 4096, 4096, 2),
    _linear_case("llama3.1-8b-wk", 2048, 4096, 1024, 2),
    _rope_case("small", 64, 4, 64, 2, quick=True),
    _rope_case("llama3.1-8b", 2048, 32, 128, 2),
]
OPS = ("attention", "rms_norm", "swiglu", "linear", "rope")


def _array_positions(args) -> list[int]:
  return [i for i, a in enumerate(args) if hasattr(a, "dtype")]


def _rel(a, b) -> float:
  a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _max_rel(xs, ys) -> float:
  return max(_rel(x, y) for x, y in zip(xs, ys))


class _Runner:
  """Runs cases under one provider at a time, with fresh traces per run."""

  def __init__(self):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    from jaxtitan import ops  # pylint: disable=g-import-not-at-top

    self.jax, self.jnp, self.ops = jax, jnp, ops

  def _op(self, case, kwargs):
    return lambda *a: getattr(self.ops, case.op)(*a, **kwargs)

  def _forward_backward(self, case, kwargs, args):
    """fb(arrays, ct) -> (output, gradients of `arrays`).

    `arrays` are the array arguments; the others (such as eps) stay static.
    """
    jax, fn = self.jax, self._op(case, kwargs)
    idx = _array_positions(args)

    def full(arrays):
      out = list(args)
      for i, a in zip(idx, arrays):
        out[i] = a
      return out

    def fb(arrays, ct):
      y, pullback = jax.vjp(lambda *a: fn(*full(a)), *arrays)
      return y, pullback(ct)
    return fb

  def traced(self, provider, build):
    """Runs `build()` (which traces) under `provider` and reports what ran.

    Returns (result, selection or None, reason it fell back or None).
    """
    from jaxtitan.ops import registry  # pylint: disable=g-import-not-at-top

    registry.reset_selections()
    with registry.policy(registry.Policy({op: (provider,) for op in OPS})):
      result = build()
    chosen = registry.selections()
    hit = [s for s in chosen if s.provider == provider]
    if hit:
      return result, hit[0], None
    reasons = [r for s in chosen for p, r in s.skipped if p == provider]
    return result, None, reasons[0] if reasons else "not dispatched"

  def inputs(self, case, rng, batch=None):
    args, kwargs = case.make(rng) if batch is None else case.make(rng, batch)
    bf16 = self.jnp.bfloat16
    args = tuple(self.jnp.asarray(a, bf16) if isinstance(a, np.ndarray) else a for a in args)
    return args, kwargs

  def f32(self, args):
    jnp = self.jnp
    return tuple(a.astype(jnp.float32) if hasattr(a, "dtype") else a for a in args)


def _check(errs, native_errs, provider, op):
  floor = TOLERANCE.get(provider, 0.0) * GEMMS.get(op, 1)
  bad = [(e, n) for e, n in zip(errs, native_errs) if e > max(2.0 * n, floor, 1e-6)]
  return not bad


def _compare(runner, case, provider, mode, rng, devices):
  """One case in one mode: a result row."""
  jax, jnp = runner.jax, runner.jnp
  row = {"op": case.op, "case": case.name, "provider": provider, "mode": mode}
  batch = 4 if mode in ("sharded", "manual", "mpmd") else None
  args, kwargs = runner.inputs(case, rng, batch)

  if mode == "mpmd":
    return {**row, **_mpmd(runner, case, provider, args, kwargs, devices)}

  y0 = jax.eval_shape(runner._op(case, kwargs), *args)
  ct = jnp.asarray(rng.standard_normal(y0.shape).astype(np.float32), y0.dtype)
  positions = _array_positions(args)

  def run(p, args, ct):
    fb = runner._forward_backward(case, kwargs, args)
    arrays = tuple(args[i] for i in positions)

    def build():
      if mode == "vmap":
        stacked = tuple(jnp.stack([a, a[::-1]]) for a in arrays)
        return jax.jit(jax.vmap(fb))(stacked, jnp.stack([ct, ct * 0.5]))
      if mode in ("sharded", "manual"):
        mesh = jax.sharding.Mesh(np.asarray(devices[:4]), ("data",))
        P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
        specs = tuple(P("data") if i in case.batched else P() for i in positions)
        placed = tuple(jax.device_put(a, jax.sharding.NamedSharding(mesh, s))
                       for a, s in zip(arrays, specs))
        ct_placed = jax.device_put(ct, jax.sharding.NamedSharding(mesh, P("data")))
        if mode == "sharded":
          return jax.jit(fb)(placed, ct_placed)

        def local(arrays, ct):
          y, grads = fb(arrays, ct)
          grads = tuple(g if s == P("data") else jax.lax.psum(g, "data")
                        for g, s in zip(grads, specs))
          return y, grads
        return jax.jit(jax.shard_map(
            local, mesh=mesh, axis_names={"data"}, check_vma=False,
            in_specs=(specs, P("data")), out_specs=(P("data"), specs)))(placed, ct_placed)
      return jax.jit(fb)(arrays, ct)
    return runner.traced(p, build)

  (y, g), sel, reason = run(provider, args, ct)
  if sel is None:
    return {**row, "status": "skip", "reason": reason}
  (yn, gn), _, _ = run("native", args, ct)
  (yr, gr), _, _ = run("native", runner.f32(args), ct.astype(jnp.float32))
  errs = [_rel(y, yr)] + [_rel(a, b) for a, b in zip(g, gr)]
  native_errs = [_rel(yn, yr)] + [_rel(a, b) for a, b in zip(gn, gr)]
  ok = _check(errs, native_errs, provider, case.op)
  row.update(status="pass" if ok else "FAIL", kernel_origin=sel.kernel_origin,
             fwd_err=errs[0], native_fwd_err=native_errs[0],
             grad_err=max(errs[1:]), native_grad_err=max(native_errs[1:]))
  if mode == "bench":
    row.update(_bench(runner, case, provider, args, kwargs, ct))
  return row


def _bench(runner, case, provider, args, kwargs, ct, iters=20):
  jax = runner.jax
  arrays = tuple(args[i] for i in _array_positions(args))
  times = {}
  for p in (provider, "native"):
    # A new function per provider: jit caches traces by function object.
    fb = runner._forward_backward(case, kwargs, args)
    f, _, _ = runner.traced(p, lambda: jax.jit(fb).lower(arrays, ct).compile())
    for _ in range(3):
      jax.block_until_ready(f(arrays, ct))
    samples = []
    for _ in range(iters):
      t0 = time.perf_counter()
      jax.block_until_ready(f(arrays, ct))
      samples.append(time.perf_counter() - t0)
    times[p] = float(np.median(samples)) * 1e3
  return {"ms": times[provider], "native_ms": times["native"],
          "speedup": times["native"] / times[provider]}


def _mpmd(runner, case, provider, args, kwargs, devices):
  """The op inside stage 0 of a 2-stage train step; SGD(1) recovers gradients."""
  jax, jnp = runner.jax, runner.jnp
  import optax  # pylint: disable=g-import-not-at-top
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name
  op = runner._op(case, kwargs)
  x = args[0]
  topology = jt_mpmd.make_topology(2, devices[:8])
  meshes = list(topology.values())
  config = jt_mpmd.make_config(topology, "ONE_FWD_ONE_BWD")
  act = NamedSharding(meshes[0], P(jt_mpmd.DATA_AXIS))

  if case.op == "attention":
    hkv = args[1].shape[2]
    p0 = {"s": jnp.ones(x.shape[-1], jnp.float32), "ks": jnp.ones(x.shape[-1], jnp.float32),
          "vs": jnp.ones(x.shape[-1], jnp.float32)}

    def stage0(p, x):
      kv = x[:, :, :hkv]
      y = op(x * p["s"].astype(x.dtype), kv * p["ks"].astype(x.dtype),
             kv * p["vs"].astype(x.dtype))
      return jax.lax.with_sharding_constraint(y, act)
  else:
    p0 = {"s": jnp.ones(x.shape[-1], jnp.float32),
          "w": tuple(a.astype(jnp.float32) for a in args[1:] if hasattr(a, "dtype"))}
    extra = tuple(a for a in args[1:] if not hasattr(a, "dtype"))

    def stage0(p, x):
      ws = tuple(w.astype(x.dtype) for w in p["w"])
      y = op(x * p["s"].astype(x.dtype), *ws, *extra)
      return jax.lax.with_sharding_constraint(y, act)

  y0 = jax.eval_shape(stage0, p0, x)
  p1 = {"w": jnp.ones(y0.shape[-1], jnp.float32)}

  def stage1(p, y, t):
    return jnp.mean((y.astype(jnp.float32) * p["w"] - t) ** 2)

  rng = np.random.default_rng(1)
  xs = jnp.stack([x, jnp.asarray(rng.standard_normal(x.shape), x.dtype)])
  ts = jnp.asarray(rng.standard_normal((2,) + y0.shape), jnp.float32)
  opt = optax.sgd(1.0)
  params = (p0, p1)
  states = tuple(opt.init(p) for p in params)

  def replicated(tree, mesh):
    return jax.tree.map(lambda _: NamedSharding(mesh, P()), tree)

  p_sh = tuple(replicated(p, m) for p, m in zip(params, meshes))
  s_sh = tuple(replicated(s, m) for s, m in zip(states, meshes))
  x_sh = NamedSharding(meshes[0], P(None, jt_mpmd.DATA_AXIS))
  t_sh = NamedSharding(meshes[1], P(None, jt_mpmd.DATA_AXIS))

  def run(p):
    def build():
      step = jt_mpmd.make_train_step([stage0, stage1], opt)
      scalar = NamedSharding(meshes[1], P())
      compiled = mpmd.jit(step, config, in_shardings=(p_sh, s_sh, x_sh, t_sh),
                          out_shardings=(p_sh, s_sh, scalar, scalar)
                          ).lower(params, states, xs, ts).compile()
      new, _, loss, _ = compiled(jax.device_put(params, p_sh), jax.device_put(states, s_sh),
                              jax.device_put(xs, x_sh), jax.device_put(ts, t_sh))
      grads = jax.tree.map(lambda a, b: np.asarray(a) - np.asarray(b), params, new)
      return float(loss), jax.tree.leaves(grads)
    return runner.traced(p, build)

  (loss, grads), sel, reason = run(provider)
  if sel is None:
    return {"status": "skip", "reason": reason}
  (loss_n, grads_n), _, _ = run("native")
  err = max([_rel(loss, loss_n)] + [_rel(a, b) for a, b in zip(grads, grads_n)])
  ok = err <= max(TOLERANCE.get(provider, 0.0) * GEMMS.get(case.op, 1), 3e-2)
  return {"status": "pass" if ok else "FAIL", "kernel_origin": sel.kernel_origin,
          "loss": loss, "native_loss": loss_n, "grad_err": err}


def _format(row) -> str:
  head = f"{row['op']:9s} {row['case']:15s} {row['provider']:11s} {row['mode']:7s} {row['status']:5s}"
  if row["status"] in ("skip", "ERROR"):
    return f"{head}  {row['reason']}"
  if row["mode"] == "mpmd":
    return (f"{head}  loss {row['loss']:.5f} vs native {row['native_loss']:.5f}, "
            f"grads within {row['grad_err']:.1e}  [{row['kernel_origin']}]")
  text = (f"{head}  fwd {row['fwd_err']:.1e} (native {row['native_fwd_err']:.1e}), "
          f"grad {row['grad_err']:.1e} (native {row['native_grad_err']:.1e})")
  if "ms" in row:
    text += f", {row['ms']:.2f} ms vs native {row['native_ms']:.2f} ms ({row['speedup']:.2f}x)"
  return f"{text}  [{row['kernel_origin']}]"


def run(providers, modes, *, quick=False, ops_filter=None, cases_filter=None, out=None) -> list[dict]:
  import jax  # pylint: disable=g-import-not-at-top

  runner = _Runner()
  devices = jax.devices()
  rows = []
  for case in CASES:
    if quick != case.quick or (ops_filter and case.op not in ops_filter):
      continue
    if cases_filter and case.name not in cases_filter:
      continue
    for provider in providers:
      for mode in modes:
        if mode in ("sharded", "manual", "mpmd") and len(devices) < (8 if mode == "mpmd" else 4):
          continue
        try:
          row = _compare(runner, case, provider, mode, np.random.default_rng(0), devices)
        except Exception as e:  # pylint: disable=broad-exception-caught
          message = str(e).strip().splitlines()
          reason = next((l for l in reversed(message) if "Error" in l), message[-1] if message else "")
          row = {"op": case.op, "case": case.name, "provider": provider, "mode": mode,
                 "status": "ERROR", "reason": f"{type(e).__name__}: {reason[:300]}"}
        rows.append(row)
        print(_format(row), flush=True)
        if out:
          out.write(json.dumps(row) + "\n")
          out.flush()
  return rows


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--platform", default="rocm", choices=["cpu", "rocm"])
  parser.add_argument("--providers", default="te,aiter")
  parser.add_argument("--modes", default="single,vmap,sharded,mpmd")
  parser.add_argument("--ops", default="", help="Comma-separated ops; default all.")
  parser.add_argument("--cases", default="", help="Comma-separated case names; default all.")
  parser.add_argument("--quick", action="store_true", help="Only the small cases.")
  parser.add_argument("--out", default="", help="Append one JSON row per check here.")
  args = parser.parse_args(argv)
  if args.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax  # pylint: disable=g-import-not-at-top

    jax.config.update("jax_num_cpu_devices", 8)
  modes = args.modes.split(",")
  for m in modes:
    if m not in MODES:
      parser.error(f"unknown mode {m}; known: {', '.join(MODES)}")
  out = open(args.out, "a") if args.out else None
  rows = run(args.providers.split(","), modes, quick=args.quick,
             ops_filter=set(filter(None, args.ops.split(","))),
             cases_filter=set(filter(None, args.cases.split(","))), out=out)
  failed = [r for r in rows if r["status"] in ("FAIL", "ERROR")]
  print(f"{len(rows)} checks: {sum(r['status'] == 'pass' for r in rows)} passed, "
        f"{len(failed)} failed, {sum(r['status'] == 'skip' for r in rows)} skipped")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
