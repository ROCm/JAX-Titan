"""Pretrains a Llama model with Shardy MPMD pipeline parallelism.

  python -m jaxtitan.trainer.train --model=llama3.1-8b --num_stages=2

`--parallel=dp` and `--parallel=fsdp` train the same model on the same batches
without the pipeline, as one `jax.jit` program over all devices.
"""

import argparse
import functools
import json
import os
import time

from jaxtitan import paths

_NORM_PARAMS = ("attn_norm", "mlp_norm", "final_norm")
_DATA_AXIS = "data"


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="llama3.1-8b")
  parser.add_argument(
      "--parallel", default="mpmd", choices=["mpmd", "dp", "fsdp"],
      help="mpmd: Shardy MPMD pipeline, data parallel inside each stage. "
      "dp, fsdp: no pipeline; one jax.jit program over all devices with "
      "replicated or sharded parameters and optimizer state.",
  )
  parser.add_argument(
      "--num_stages", type=int, default=2,
      help="Pipeline stages. dp and fsdp start from the same weights as the "
      "pipeline with this many stages.",
  )
  parser.add_argument(
      "--schedule", default="ONE_FWD_ONE_BWD",
      help="Shardy pipeline schedule name, or empty for the default order.",
  )
  parser.add_argument("--microbatches", type=int, default=8)
  parser.add_argument(
      "--microbatch_size", type=int, default=4,
      help="Sequences per microbatch, sharded across a stage's devices. dp and "
      "fsdp run all microbatches as one batch.",
  )
  parser.add_argument("--seq_len", type=int, default=2048)
  parser.add_argument("--steps", type=int, default=50)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--warmup_steps", type=int, default=10)
  parser.add_argument(
      "--data",
      default=str(paths.work_dir() / "data/wikitext103_train.llama3.bin"),
      help="uint32 token file, or 'synthetic'.",
  )
  parser.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      "--peak_tflops_per_device", type=float, default=2500.0,
      help="Dense BF16 peak used for MFU; 2500 for MI355X.",
  )
  parser.add_argument("--log_file", default="")
  parser.add_argument("--platform", default="", help="'cpu' runs on 8 CPU devices.")
  parser.add_argument("--seed", type=int, default=0)
  return parser.parse_args(argv)


def _decay_mask(params):
  import jax  # pylint: disable=g-import-not-at-top

  return jax.tree_util.tree_map_with_path(
      lambda path, _: not any(getattr(k, "key", None) in _NORM_PARAMS for k in path),
      params,
  )


def _build_mpmd(args, cfg, devices, optimizer):
  """Shardy MPMD pipeline: one mesh per stage, data parallel inside each."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  num_stages = args.num_stages
  topology = jt_mpmd.make_topology(num_stages, devices)
  config = jt_mpmd.make_config(topology, args.schedule or None)
  meshes = [topology[jt_mpmd.mesh_name(i)] for i in range(num_stages)]
  replicated = [NamedSharding(m, P()) for m in meshes]
  activation_sharding = NamedSharding(meshes[0], P(jt_mpmd.DATA_AXIS))

  def constrain(h):
    return jax.lax.with_sharding_constraint(h, activation_sharding)

  stage_fns = []
  for i in range(num_stages):
    fn = functools.partial(
        llama.stage_forward, cfg=cfg, stage=i, num_stages=num_stages,
        remat=args.remat, constrain=constrain,
    )
    if i < num_stages - 1:
      stage_fns.append(functools.partial(lambda f, p, x: f(p, x, None), fn))
    else:
      stage_fns.append(fn)

  keys = jax.random.split(jax.random.key(args.seed), num_stages)
  params = tuple(
      jax.jit(
          functools.partial(llama.init_stage_params, cfg, stage=i, num_stages=num_stages),
          out_shardings=replicated[i],
      )(keys[i])
      for i in range(num_stages)
  )
  opt_states = tuple(
      jax.jit(optimizer.init, out_shardings=replicated[i])(params[i])
      for i in range(num_stages)
  )

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  param_shardings = tuple(
      jax.tree.map(lambda _, s=replicated[i]: s, params[i]) for i in range(num_stages)
  )
  opt_shardings = tuple(
      jax.tree.map(lambda _, s=replicated[i]: s, opt_states[i])
      for i in range(num_stages)
  )
  token_sharding = NamedSharding(meshes[0], P(None, jt_mpmd.DATA_AXIS, None))
  target_sharding = NamedSharding(meshes[-1], P(None, jt_mpmd.DATA_AXIS, None))
  loss_sharding = NamedSharding(meshes[-1], P())

  jitted = mpmd.jit(
      jt_mpmd.make_train_step(stage_fns, optimizer),
      config,
      in_shardings=(param_shardings, opt_shardings, token_sharding, target_sharding),
      out_shardings=(param_shardings, opt_shardings, loss_sharding),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    batch = batch.reshape(num_mb, mb, seq + 1)
    return (
        jax.device_put(batch[..., :-1], token_sharding),
        jax.device_put(batch[..., 1:], target_sharding),
    )

  batch_spec = jax.ShapeDtypeStruct((num_mb, mb, seq), jnp.int32)
  return params, opt_states, jitted, batch_spec, put_batch


def _fsdp_spec(path, shape, num_devices):
  """Shards the first dimension that splits evenly, never the layer axis."""
  import jax  # pylint: disable=g-import-not-at-top

  first = 1 if any(getattr(k, "key", None) == "layers" for k in path) else 0
  for dim in range(first, len(shape)):
    if shape[dim] % num_devices == 0:
      return jax.sharding.PartitionSpec(*([None] * dim + [_DATA_AXIS]))
  return jax.sharding.PartitionSpec()


def _build_spmd(args, cfg, devices, optimizer):
  """No pipeline: one `jax.jit` program, data parallel over all devices."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  mesh = jax.sharding.Mesh(np.asarray(devices), (_DATA_AXIS,))
  batch_sharding = NamedSharding(mesh, P(_DATA_AXIS))

  def sharding(path, leaf):
    if args.parallel == "fsdp":
      return NamedSharding(mesh, _fsdp_spec(path, leaf.shape, len(devices)))
    return NamedSharding(mesh, P())

  def init(keys):
    return llama.merge_stage_params([
        llama.init_stage_params(cfg, keys[i], stage=i, num_stages=args.num_stages)
        for i in range(args.num_stages)
    ])

  keys = jax.random.split(jax.random.key(args.seed), args.num_stages)
  param_shardings = jax.tree_util.tree_map_with_path(sharding, jax.eval_shape(init, keys))
  params = jax.jit(init, out_shardings=param_shardings)(keys)
  opt_shardings = jax.tree_util.tree_map_with_path(
      sharding, jax.eval_shape(optimizer.init, params)
  )
  opt_state = jax.jit(optimizer.init, out_shardings=opt_shardings)(params)

  loss_fn = functools.partial(
      llama.stage_forward, cfg=cfg, stage=0, num_stages=1, remat=args.remat,
      constrain=lambda h: jax.lax.with_sharding_constraint(h, batch_sharding),
  )

  def train_step(params, opt_state, tokens, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

  jitted = jax.jit(
      train_step,
      in_shardings=(param_shardings, opt_shardings, batch_sharding, batch_sharding),
      out_shardings=(param_shardings, opt_shardings, NamedSharding(mesh, P())),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    return (
        jax.device_put(batch[:, :-1], batch_sharding),
        jax.device_put(batch[:, 1:], batch_sharding),
    )

  batch = args.microbatches * args.microbatch_size
  batch_spec = jax.ShapeDtypeStruct((batch, args.seq_len), jnp.int32)
  return params, opt_state, jitted, batch_spec, put_batch


def main(argv=None):
  args = parse_args(argv)
  if args.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top

  if args.platform == "cpu":
    jax.config.update("jax_num_cpu_devices", 8)

  import jaxtitan  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import
  from jaxtitan.data.text import WindowSampler  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top

  cfg = llama.FLAVORS[args.model]
  devices = jax.devices()
  if args.parallel == "mpmd":
    layout = f"{args.num_stages} stages | schedule={args.schedule or 'default'}"
  else:
    layout = f"{args.parallel} over {len(devices)} devices, no pipeline"
  print(
      f"jax {jax.__version__} | {len(devices)} x {devices[0].device_kind} | "
      f"{args.model} ({cfg.num_params() / 1e9:.2f}B params) | {layout}",
      flush=True,
  )

  lr = optax.warmup_cosine_decay_schedule(
      0.0, args.lr, args.warmup_steps, max(args.steps, args.warmup_steps + 1),
      end_value=args.lr * 0.1,
  )
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1, mask=_decay_mask),
  )

  build = _build_mpmd if args.parallel == "mpmd" else _build_spmd
  t0 = time.time()
  params, opt_states, jitted, batch_spec, put_batch = build(args, cfg, devices, optimizer)
  print(f"initialized params and optimizer state in {time.time() - t0:.1f}s", flush=True)

  t0 = time.time()
  lowered = jitted.lower(params, opt_states, batch_spec, batch_spec)
  t1 = time.time()
  compiled = lowered.compile()
  print(
      f"lowered in {t1 - t0:.1f}s, compiled in {time.time() - t1:.1f}s",
      flush=True,
  )

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  if args.data == "synthetic":
    rng = np.random.default_rng(args.seed)

    def next_batch():
      return rng.integers(0, cfg.vocab_size, (num_mb * mb, seq + 1), dtype=np.int32)

  else:
    sampler = WindowSampler(args.data, seq, seed=args.seed)
    print(f"data: {args.data} ({sampler.num_tokens / 1e6:.1f}M tokens)", flush=True)

    def next_batch():
      return sampler.next(num_mb * mb)

  tokens_per_step = num_mb * mb * seq
  flops_per_step = cfg.flops_per_token(seq) * tokens_per_step
  peak_flops = len(devices) * args.peak_tflops_per_device * 1e12
  log = open(args.log_file, "a") if args.log_file else None

  for step in range(args.steps):
    tokens, targets = put_batch(next_batch())
    t0 = time.time()
    params, opt_states, loss = compiled(params, opt_states, tokens, targets)
    loss = float(loss)
    dt = time.time() - t0
    record = {
        "step": step,
        "loss": loss,
        "step_time_s": dt,
        "tokens_per_s": tokens_per_step / dt,
        "mfu": flops_per_step / dt / peak_flops,
    }
    print(
        f"step {step:4d} | loss {loss:8.4f} | {dt * 1e3:8.1f} ms | "
        f"{record['tokens_per_s']:10.0f} tok/s | MFU {100 * record['mfu']:5.1f}%",
        flush=True,
    )
    if log:
      log.write(json.dumps(record) + "\n")
      log.flush()
    if step == 0:
      peaks = [
          (d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30
          for d in devices
      ]
      print("peak HBM per device (GiB): " + " ".join(f"{p:.1f}" for p in peaks), flush=True)
    if not np.isfinite(loss):
      raise FloatingPointError(f"non-finite loss at step {step}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
