"""Checks MaxDiffusion FLUX.1 stages against MaxDiffusion and single-mesh jax.jit.

Builds a small FLUX.1 from MaxDiffusion's transformer and checks that
1. the JAX-Titan stages initialize to the transformer's weights and predict
   the same velocity, and
2. the MPMD pipeline step over the stages, with MaxDiffusion's flow-matching
   loss, matches the same step as one single-mesh `jax.jit` program, in loss
   and parameters.

Needs MaxDiffusion with the patches in third_party/patches (see
docs/getting_started.md).

  python examples/maxdiffusion_flux_parity.py --platform=cpu
"""

import argparse
import os
import sys


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", default="cpu", choices=["cpu", "rocm"])
  parser.add_argument("--flavor", default="flux-dev", choices=["flux-dev", "flux-schnell"])
  parser.add_argument("--schedule", default="ONE_FWD_ONE_BWD")
  parser.add_argument("--num_stages", type=int, default=2)
  parser.add_argument("--blocks", default="",
                      help="Blocks per stage, double-stream blocks first, e.g. 3,1")
  parser.add_argument("--microbatches", type=int, default=4)
  parser.add_argument("--steps", type=int, default=3)
  parser.add_argument("--lr", type=float, default=0.05)
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

  from flax import linen as nn  # pylint: disable=g-import-not-at-top
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan.models.adapters import maxdiffusion as md  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  num_stages, num_mb, mb = args.num_stages, args.microbatches, 4
  blocks = tuple(int(n) for n in args.blocks.split(",")) if args.blocks else None
  flux = md.transformer(
      args.flavor, in_channels=16, num_layers=2, num_single_layers=2,
      attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
      pooled_projection_dim=16, axes_dims_rope=(4, 6, 6), dtype=jnp.float32,
      weights_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST,
  )
  model = md.Model(flux, num_stages=num_stages, blocks=blocks)
  stage_fns = model.stage_fns()
  host_params = jax.device_get(model.init(seed=0))

  side, text_tokens = 4, 8
  img_ids, txt_ids = md.position_ids(side, text_tokens)
  rng = np.random.default_rng(0)

  def example(*lead):
    """Model inputs and velocity targets, with leading dims `lead`."""
    b = lead + (mb,)
    x = {
        "img": rng.normal(size=b + (side * side, flux.in_channels)),
        "txt": rng.normal(size=b + (text_tokens, flux.joint_attention_dim)),
        "pooled": rng.normal(size=b + (flux.pooled_projection_dim,)),
        "timestep": rng.uniform(size=b),
        "img_ids": np.broadcast_to(img_ids, lead + img_ids.shape),
        "txt_ids": np.broadcast_to(txt_ids, lead + txt_ids.shape),
    }
    if flux.guidance_embeds:
      x["guidance"] = np.full(b, 3.5)
    target = rng.normal(size=b + (side * side, flux.in_channels))
    return jax.tree.map(lambda a: np.asarray(a, np.float32), (x, target))

  # 1. Same weights as MaxDiffusion's transformer, and the same prediction.
  x, _ = example()

  def maxdiffusion_call(fn, p, x):
    return fn(p, x["img"], x["txt"], x["pooled"], x["timestep"], x["img_ids"],
              x["txt_ids"], x.get("guidance"))

  full_params = nn.meta.unbox(jax.jit(lambda x: maxdiffusion_call(
      lambda key, *a: flux.init(key, *a), jax.random.key(0), x))(x)["params"])
  split_params = jax.device_get(model.split(full_params))
  for a, b in zip(jax.tree.leaves(split_params), jax.tree.leaves(host_params)):
    np.testing.assert_array_equal(a, b)
  ref = jax.jit(lambda p, x: maxdiffusion_call(
      lambda p, *a: flux.apply({"params": p}, *a).sample, p, x))(full_params, x)

  def chain(stage_params, x):
    for i in range(num_stages):
      x = model.forward(i, stage_params[i], x)
    return x

  pred = jax.jit(chain)(split_params, x)
  np.testing.assert_allclose(np.asarray(pred), np.asarray(ref), rtol=1e-5, atol=1e-5)
  print(f"{num_stages} MaxDiffusion {args.flavor} stages {model.blocks} match the "
        f"transformer: max |velocity diff| {float(np.max(np.abs(pred - ref))):.2e}")

  # 2. The MPMD pipeline step matches single-mesh jax.jit.
  optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(args.lr))
  host_opt = tuple(jax.device_get(optimizer.init(p)) for p in host_params)
  batches = [example(num_mb) for _ in range(args.steps)]
  topology = jt_mpmd.make_topology(num_stages)
  meshes = list(topology.values())
  config = jt_mpmd.make_config(topology, args.schedule or None)

  def reference_step(stage_params, opt_states, inputs, targets):
    def loss_fn(stage_params):
      total = 0.0
      for m in range(num_mb):
        x = jax.tree.map(lambda a, m=m: a[m], inputs)
        for i, fn in enumerate(stage_fns):
          x = fn(stage_params[i], x, targets[m]) if i == num_stages - 1 else fn(stage_params[i], x)
        total = total + x
      return total / num_mb

    loss, grads = jax.value_and_grad(loss_fn)(stage_params)
    new_p, new_s = [], []
    for p, g, s in zip(stage_params, grads, opt_states):
      u, s = optimizer.update(g, s, p)
      new_p.append(optax.apply_updates(p, u))
      new_s.append(s)
    return tuple(new_p), tuple(new_s), loss

  def replicated(tree, mesh):
    return jax.tree.map(lambda _: NamedSharding(mesh, P()), tree)

  param_sh = tuple(replicated(p, m) for p, m in zip(host_params, meshes))
  opt_sh = tuple(replicated(s, m) for s, m in zip(host_opt, meshes))
  in_sh = {
      k: NamedSharding(meshes[0], P() if k.endswith("_ids") else P(None, "data"))
      for k in batches[0][0]
  }
  tgt_sh = NamedSharding(meshes[-1], P(None, "data"))
  train_step = jt_mpmd.make_train_step(stage_fns, optimizer)
  compiled = mpmd.jit(
      train_step, config,
      in_shardings=(param_sh, opt_sh, in_sh, tgt_sh),
      out_shardings=(param_sh, opt_sh, NamedSharding(meshes[-1], P())),
  ).lower(host_params, host_opt, *batches[0]).compile()
  ref_step = jax.jit(reference_step)

  mpmd_params, mpmd_opt = jax.device_put((host_params, host_opt), (param_sh, opt_sh))
  ref_params, ref_opt = host_params, host_opt
  for step, (inputs, targets) in enumerate(batches):
    mpmd_params, mpmd_opt, mpmd_loss = compiled(
        mpmd_params, mpmd_opt, jax.device_put(inputs, in_sh), jax.device_put(targets, tgt_sh)
    )
    ref_params, ref_opt, ref_loss = ref_step(ref_params, ref_opt, inputs, targets)
    print(f"step {step}: mpmd {float(mpmd_loss):.6f}  ref {float(ref_loss):.6f}")
    np.testing.assert_allclose(float(mpmd_loss), float(ref_loss), rtol=1e-5)
  for a, b in zip(jax.tree.leaves(jax.device_get(mpmd_params)),
                  jax.tree.leaves(jax.device_get(ref_params))):
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
  print(f"MPMD pipeline of MaxDiffusion {args.flavor} stages {model.blocks} "
        f"({args.schedule}) matches the single-mesh reference")
  return 0


if __name__ == "__main__":
  sys.exit(main())
