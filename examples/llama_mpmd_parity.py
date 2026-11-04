"""Checks the Llama MPMD pipeline step against a single-mesh `jax.jit` step.

Both steps use the same stage functions, microbatches and per-stage optimizer,
in fp32, so their losses and parameters must agree. SGD keeps the parameter
check tight: Adam magnifies rounding noise in near-zero embedding gradients.

  python examples/llama_mpmd_parity.py --platform=cpu
"""

import argparse
import functools
import os
import sys


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", default="cpu", choices=["cpu", "rocm"])
  parser.add_argument("--schedule", default="ONE_FWD_ONE_BWD")
  parser.add_argument("--steps", type=int, default=4)
  args = parser.parse_args()
  if args.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top

  if args.platform == "cpu":
    jax.config.update("jax_num_cpu_devices", 8)

  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  cfg = llama.FLAVORS["llama3-test"]
  num_stages, num_mb, mb, seq = 2, 4, 4, 64
  topology = jt_mpmd.make_topology(num_stages)
  config = jt_mpmd.make_config(topology, args.schedule or None)
  meshes = [topology[jt_mpmd.mesh_name(i)] for i in range(num_stages)]
  replicated = [NamedSharding(m, P()) for m in meshes]

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

  param_sh = tuple(
      jax.tree.map(lambda _, s=replicated[i]: s, host_params[i])
      for i in range(num_stages)
  )
  opt_sh = tuple(
      jax.tree.map(lambda _, s=replicated[i]: s, host_opt[i])
      for i in range(num_stages)
  )
  tok_sh = NamedSharding(meshes[0], P(None, jt_mpmd.DATA_AXIS, None))
  tgt_sh = NamedSharding(meshes[-1], P(None, jt_mpmd.DATA_AXIS, None))
  compiled = mpmd.jit(
      jt_mpmd.make_train_step(stage_fns, optimizer), config,
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
  print(f"MPMD pipeline ({args.schedule}) matches the single-mesh reference")
  return 0


if __name__ == "__main__":
  sys.exit(main())
