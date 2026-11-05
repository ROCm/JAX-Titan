"""Checks the Llama MPMD pipeline step against a single-mesh `jax.jit` step.

Both steps use the same stage functions, microbatches and per-stage optimizer,
in fp32, so their losses and parameters must agree. SGD keeps the parameter
check tight: Adam magnifies rounding noise in near-zero embedding gradients.

  python examples/llama_mpmd_parity.py --platform=cpu
  python examples/llama_mpmd_parity.py --platform=cpu --stage_fsdp --tensor_parallel=2
"""

import argparse
import functools
import json
import os
import sys


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", default="cpu", choices=["cpu", "rocm"])
  parser.add_argument("--schedule", default="ONE_FWD_ONE_BWD")
  parser.add_argument("--mpmd_options", default="", help="JSON object, as in the trainer.")
  parser.add_argument("--num_stages", type=int, default=2)
  parser.add_argument("--stages_per_mesh", type=int, default=1)
  parser.add_argument("--microbatches", type=int, default=4)
  parser.add_argument("--steps", type=int, default=4)
  parser.add_argument("--reduce_grads", default="step", choices=["microbatch", "step"])
  parser.add_argument("--stage_fsdp", action="store_true")
  parser.add_argument("--tensor_parallel", type=int, default=1)
  return parser.parse_args(argv)


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

  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan import parallel  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  cfg = llama.FLAVORS["llama3-test"]
  num_stages, num_mb, mb, seq = args.num_stages, args.microbatches, 4, 64
  num_meshes = num_stages // args.stages_per_mesh
  topology = jt_mpmd.make_topology(num_meshes, tensor_parallel=args.tensor_parallel)
  options = json.loads(args.mpmd_options) if args.mpmd_options else None
  config = jt_mpmd.make_config(
      topology, args.schedule or None, options, num_stages=num_stages
  )
  meshes = [topology[jt_mpmd.mesh_name(i % num_meshes)] for i in range(num_stages)]

  stage_fns = []
  for i in range(num_stages):
    fn = functools.partial(
        llama.stage_forward, cfg=cfg, stage=i, num_stages=num_stages,
        remat=True, compute_dtype=jnp.float32,
    )
    if i < num_stages - 1:
      stage_fns.append(functools.partial(lambda f, p, x: f(p, x, None), fn))
    else:
      stage_fns.append(fn)
  optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(0.5))

  keys = jax.random.split(jax.random.key(0), num_stages)
  host_params = tuple(
      jax.device_get(llama.init_stage_params(cfg, keys[i], i, num_stages))
      for i in range(num_stages)
  )
  host_opt = tuple(jax.device_get(optimizer.init(p)) for p in host_params)
  rng = np.random.default_rng(0)
  batches = [
      rng.integers(0, cfg.vocab_size, (num_mb, mb, seq + 1), dtype=np.int32)
      for _ in range(args.steps)
  ]

  def reference_step(stage_params, opt_states, tokens, targets):
    def loss_fn(stage_params):
      total = 0.0
      for m in range(num_mb):
        x = tokens[m]
        for i, fn in enumerate(stage_fns):
          x = fn(stage_params[i], x, targets[m]) if i == num_stages - 1 else fn(
              stage_params[i], x)
        total = total + x
      return total / num_mb

    loss, grads = jax.value_and_grad(loss_fn)(stage_params)
    new_p, new_s = [], []
    for p, g, s in zip(stage_params, grads, opt_states):
      u, s = optimizer.update(g, s, p)
      new_p.append(optax.apply_updates(p, u))
      new_s.append(s)
    return tuple(new_p), tuple(new_s), loss

  specs = [
      parallel.tree_specs(
          (host_params[i], host_opt[i]), llama.logical_axes, meshes[i].shape,
          fsdp=args.stage_fsdp,
      )
      for i in range(num_stages)
  ]
  param_sh = tuple(
      jax.tree.map(lambda s, m=meshes[i]: NamedSharding(m, s), specs[i][0])
      for i in range(num_stages)
  )
  opt_sh = tuple(
      jax.tree.map(lambda s, m=meshes[i]: NamedSharding(m, s), specs[i][1])
      for i in range(num_stages)
  )
  tok_sh = NamedSharding(meshes[0], P(None, jt_mpmd.DATA_AXIS, None))
  tgt_sh = NamedSharding(meshes[-1], P(None, jt_mpmd.DATA_AXIS, None))
  copy_axes = None
  if args.reduce_grads == "step":
    copy_axes = [llama.device_copy_axes(p) for p in host_params]
  train_step = jt_mpmd.make_train_step(
      stage_fns, optimizer, copy_axes=copy_axes,
      param_specs=[s[0] for s in specs], mesh=meshes[0],
  )
  compiled = mpmd.jit(
      train_step, config,
      in_shardings=(param_sh, opt_sh, tok_sh, tgt_sh),
      out_shardings=(param_sh, opt_sh, NamedSharding(meshes[-1], P())),
  ).lower(host_params, host_opt, batches[0][..., :-1], batches[0][..., 1:]).compile()
  ref = jax.jit(reference_step)

  mpmd_params, mpmd_opt = jax.device_put((host_params, host_opt), (param_sh, opt_sh))
  ref_params, ref_opt = host_params, host_opt
  for step, batch in enumerate(batches):
    tok, tgt = batch[..., :-1], batch[..., 1:]
    mpmd_params, mpmd_opt, mpmd_loss = compiled(
        mpmd_params, mpmd_opt, jax.device_put(tok, tok_sh), jax.device_put(tgt, tgt_sh)
    )
    ref_params, ref_opt, ref_loss = ref(ref_params, ref_opt, tok, tgt)
    print(f"step {step}: mpmd {float(mpmd_loss):.6f}  ref {float(ref_loss):.6f}")
    np.testing.assert_allclose(float(mpmd_loss), float(ref_loss), rtol=1e-5)
  for a, b in zip(jax.tree.leaves(jax.device_get(mpmd_params)),
                  jax.tree.leaves(jax.device_get(ref_params))):
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
  layout = f"{num_stages} stages"
  if args.stage_fsdp:
    layout += ", FSDP"
  if args.tensor_parallel > 1:
    layout += f", tensor parallel {args.tensor_parallel}"
  print(
      f"MPMD pipeline ({args.schedule}, {layout}, gradients reduced per "
      f"{args.reduce_grads}) matches the single-mesh reference"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
