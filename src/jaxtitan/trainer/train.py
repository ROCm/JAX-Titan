"""Pretrains a Llama model with Shardy MPMD pipeline parallelism.

  python -m jaxtitan.trainer.train --model=llama3.1-8b --num_stages=2
"""

import argparse
import functools
import json
import os
import time

from jaxtitan import paths

_NORM_PARAMS = ("attn_norm", "mlp_norm", "final_norm")


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="llama3.1-8b")
  parser.add_argument("--num_stages", type=int, default=2)
  parser.add_argument(
      "--schedule", default="ONE_FWD_ONE_BWD",
      help="Shardy pipeline schedule name, or empty for the default order.",
  )
  parser.add_argument("--microbatches", type=int, default=8)
  parser.add_argument(
      "--microbatch_size", type=int, default=4,
      help="Sequences per microbatch, sharded across a stage's devices.",
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


def main(argv=None):
  args = parse_args(argv)
  if args.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top

  if args.platform == "cpu":
    jax.config.update("jax_num_cpu_devices", 8)

  import jaxtitan  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan.data.text import WindowSampler  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  cfg = llama.FLAVORS[args.model]
  num_stages = args.num_stages
  devices = jax.devices()
  print(
      f"jax {jax.__version__} | {len(devices)} x {devices[0].device_kind} | "
      f"{args.model} ({cfg.num_params() / 1e9:.2f}B params) | "
      f"{num_stages} stages | schedule={args.schedule or 'default'}",
      flush=True,
  )

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

  lr = optax.warmup_cosine_decay_schedule(
      0.0, args.lr, args.warmup_steps, max(args.steps, args.warmup_steps + 1),
      end_value=args.lr * 0.1,
  )
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1, mask=_decay_mask),
  )

  t0 = time.time()
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
  print(f"initialized params and optimizer state in {time.time() - t0:.1f}s", flush=True)

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
  batch_spec = jax.ShapeDtypeStruct((num_mb, mb, seq), jnp.int32)

  train_step = jt_mpmd.make_train_step(stage_fns, optimizer)
  t0 = time.time()
  lowered = mpmd.jit(
      train_step,
      config,
      in_shardings=(param_shardings, opt_shardings, token_sharding, target_sharding),
      out_shardings=(param_shardings, opt_shardings, loss_sharding),
      donate_argnums=(0, 1),
  ).lower(params, opt_states, batch_spec, batch_spec)
  t1 = time.time()
  compiled = lowered.compile()
  print(
      f"lowered in {t1 - t0:.1f}s, compiled in {time.time() - t1:.1f}s",
      flush=True,
  )

  if args.data == "synthetic":
    rng = np.random.default_rng(args.seed)

    def next_batch():
      return rng.integers(0, cfg.vocab_size, (num_mb, mb, seq + 1), dtype=np.int32)

  else:
    sampler = WindowSampler(args.data, seq, seed=args.seed)
    print(f"data: {args.data} ({sampler.num_tokens / 1e6:.1f}M tokens)", flush=True)

    def next_batch():
      return sampler.next(num_mb * mb).reshape(num_mb, mb, seq + 1)

  tokens_per_step = num_mb * mb * seq
  flops_per_step = cfg.flops_per_token(seq) * tokens_per_step
  peak_flops = len(devices) * args.peak_tflops_per_device * 1e12
  log = open(args.log_file, "a") if args.log_file else None

  for step in range(args.steps):
    batch = next_batch()
    tokens = jax.device_put(batch[..., :-1], token_sharding)
    targets = jax.device_put(batch[..., 1:], target_sharding)
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
