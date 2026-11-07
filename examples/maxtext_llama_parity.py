"""Checks MaxText Llama stages against MaxText, our Llama and single-mesh jax.jit.

Builds a small Llama 3.1 from MaxText's `llama3.1-8b` config and checks that
1. the JAX-Titan stages initialize to MaxText's `Transformer` weights and give
   the same logits,
2. `jaxtitan.models.llama`, given the same weights in its layout, gives the
   same activations at every stage boundary and the same loss, and
3. the MPMD pipeline step over the stages matches the same step as one
   single-mesh `jax.jit` program, in loss and parameters.

Needs the `maxtext` extra (see docs/getting_started.md).

  python examples/maxtext_llama_parity.py --platform=cpu
"""

import argparse
import os
import sys


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", default="cpu", choices=["cpu", "rocm"])
  parser.add_argument("--model", default="llama3.1-8b")
  parser.add_argument("--schedule", default="ONE_FWD_ONE_BWD")
  parser.add_argument("--num_stages", type=int, default=2)
  parser.add_argument("--stage_layers", default="",
                      help="Decoder layers per stage, e.g. 3,1; default 2 per stage")
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
  from flax import nnx  # pylint: disable=g-import-not-at-top
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan import ops  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top
  from jaxtitan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top
  from maxtext.models import models  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  num_stages, num_mb, mb, seq = args.num_stages, args.microbatches, 4, 32
  if args.stage_layers:
    stage_layers = tuple(int(n) for n in args.stage_layers.split(","))
  else:
    stage_layers = (2,) * num_stages
  cfg = mt.config(
      args.model,
      base_emb_dim=64, base_mlp_dim=128, base_num_query_heads=4, base_num_kv_heads=2,
      head_dim=16, base_num_decoder_layers=sum(stage_layers), vocab_size=256,
      max_target_length=seq, per_device_batch_size=1,
      dtype="float32", weight_dtype="float32", matmul_precision="highest",
  )
  devices = jax.devices()
  per_stage = len(devices) // num_stages
  meshes = [
      mt.mesh(cfg, devices[i * per_stage:(i + 1) * per_stage]) for i in range(num_stages)
  ]
  model = mt.Model(cfg, meshes[0], num_stages=num_stages, layers=stage_layers)
  stage_fns = model.stage_fns()
  host_params = jax.device_get(model.init(seed=0))

  # 1. Same weights as MaxText's Transformer, and the same logits.
  with nn.logical_axis_rules(cfg.logical_axis_rules):
    transformer = models.Transformer(
        cfg, meshes[0], quant=None, rngs=nnx.Rngs(params=0, dropout=0)
    )
  graph, maxtext_params, maxtext_rest = nnx.split(transformer, nnx.Param, ...)
  split_params = jax.device_get(model.split(maxtext_params.to_pure_dict()))
  for a, b in zip(jax.tree.leaves(split_params), jax.tree.leaves(host_params)):
    np.testing.assert_array_equal(a, b)

  rng = np.random.default_rng(0)
  tokens = rng.integers(0, cfg.vocab_size, (mb, seq), dtype=np.int32)
  positions = np.broadcast_to(np.arange(seq, dtype=np.int32), (mb, seq))
  segments = np.ones((mb, seq), np.int32)

  @jax.jit
  def maxtext_logits(params, rest, t):
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      return nnx.merge(graph, params, rest)(t, positions, segments, enable_dropout=False)

  ref_logits = maxtext_logits(maxtext_params, maxtext_rest, tokens)

  def chain(stage_params, x):
    for i in range(num_stages):
      x = model.forward(i, stage_params[i], x)
    return x

  logits = jax.jit(chain)(split_params, tokens)
  np.testing.assert_allclose(np.asarray(logits), np.asarray(ref_logits), rtol=1e-5, atol=1e-5)
  print(f"{num_stages} MaxText stages match MaxText's Transformer: "
        f"max |logits diff| {float(np.max(np.abs(np.asarray(logits) - np.asarray(ref_logits)))):.2e}")

  # 1b. The same, with attention through jaxtitan.ops (native JAX).
  routed = mt.Model(cfg, meshes[0], num_stages=num_stages, layers=stage_layers,
                    attention=ops.attention)

  def routed_chain(stage_params, x):
    for i in range(num_stages):
      x = routed.forward(i, stage_params[i], x)
    return x

  ops.reset_selections()
  routed_logits = jax.jit(routed_chain)(split_params, tokens)
  assert any(s.op == "attention" for s in ops.selections()), "attention was not routed"
  np.testing.assert_allclose(np.asarray(routed_logits), np.asarray(ref_logits), rtol=1e-5,
                             atol=1e-5)
  print("with attention through jaxtitan.ops, they still match: max |logits diff| "
        f"{float(np.max(np.abs(np.asarray(routed_logits) - np.asarray(ref_logits)))):.2e}")

  # 2. jaxtitan.models.llama gives the same stage outputs with the same weights.
  llama_cfg = mt.llama_config(cfg)
  llama_params = mt.llama_params(cfg, host_params)
  targets = rng.integers(0, cfg.vocab_size, (mb, seq), dtype=np.int32)

  def stage_outputs(fns, stage_params, x, t):
    outs = []
    for i, fn in enumerate(fns):
      x = fn(stage_params[i], x, t) if i == num_stages - 1 else fn(stage_params[i], x)
      outs.append(x)
    return outs

  llama_fns = [
      lambda p, x, t=None, i=i: llama.stage_forward(
          p, x, t, cfg=llama_cfg, stage=i, num_stages=num_stages, remat=False,
          compute_dtype=jnp.float32)
      for i in range(num_stages)
  ]
  ours = jax.jit(lambda p, x, t: stage_outputs(llama_fns, p, x, t))(llama_params, tokens, targets)
  theirs = jax.jit(lambda p, x, t: stage_outputs(stage_fns, p, x, t))(host_params, tokens, targets)
  for i, (a, b) in enumerate(zip(ours, theirs)):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5,
                               err_msg=f"stage {i}")
  print(f"jaxtitan.models.llama matches the MaxText stages ({llama_cfg.rope_scaling}): "
        f"max |activation diff| "
        f"{max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(ours, theirs)):.2e}")

  # 3. The MPMD pipeline step matches single-mesh jax.jit.
  optimizer = optax.sgd(args.lr)
  max_grad_norm = 1.0
  host_opt = tuple(jax.device_get(optimizer.init(p)) for p in host_params)
  batches = [
      rng.integers(0, cfg.vocab_size, (num_mb, mb, seq + 1), dtype=np.int32)
      for _ in range(args.steps)
  ]
  topology = {jt_mpmd.mesh_name(i): m for i, m in enumerate(meshes)}
  config = jt_mpmd.make_config(topology, args.schedule or None)

  def reference_step(stage_params, opt_states, tokens, targets):
    def loss_fn(stage_params):
      total = 0.0
      for m in range(num_mb):
        x = tokens[m]
        for i, fn in enumerate(stage_fns):
          x = fn(stage_params[i], x, targets[m]) if i == num_stages - 1 else fn(stage_params[i], x)
        total = total + x
      return total / num_mb

    loss, grads = jax.value_and_grad(loss_fn)(stage_params)
    norm = jt_mpmd.global_grad_norm(grads)
    new_p, new_s = [], []
    for p, g, s in zip(stage_params, grads, opt_states):
      p, s = jt_mpmd.clipped_update(optimizer, p, g, s, norm, max_grad_norm)
      new_p.append(p)
      new_s.append(s)
    return tuple(new_p), tuple(new_s), loss, norm

  param_sh = tuple(
      jax.tree.map(lambda _, m=meshes[i]: NamedSharding(m, P()), host_params[i])
      for i in range(num_stages)
  )
  opt_sh = tuple(
      jax.tree.map(lambda _, m=meshes[i]: NamedSharding(m, P()), host_opt[i])
      for i in range(num_stages)
  )
  tok_sh = NamedSharding(meshes[0], P(None, "data", None))
  tgt_sh = NamedSharding(meshes[-1], P(None, "data", None))
  train_step = jt_mpmd.make_train_step(
      stage_fns, optimizer, max_grad_norm=max_grad_norm, mesh=meshes[0])
  scalar = NamedSharding(meshes[-1], P())
  compiled = mpmd.jit(
      train_step, config,
      in_shardings=(param_sh, opt_sh, tok_sh, tgt_sh),
      out_shardings=(param_sh, opt_sh, scalar, scalar),
  ).lower(host_params, host_opt, batches[0][..., :-1], batches[0][..., 1:]).compile()
  ref = jax.jit(reference_step)

  mpmd_params, mpmd_opt = jax.device_put((host_params, host_opt), (param_sh, opt_sh))
  ref_params, ref_opt = host_params, host_opt
  for step, batch in enumerate(batches):
    tok, tgt = batch[..., :-1], batch[..., 1:]
    mpmd_params, mpmd_opt, mpmd_loss, mpmd_norm = compiled(
        mpmd_params, mpmd_opt, jax.device_put(tok, tok_sh), jax.device_put(tgt, tgt_sh)
    )
    ref_params, ref_opt, ref_loss, ref_norm = ref(ref_params, ref_opt, tok, tgt)
    print(f"step {step}: mpmd {float(mpmd_loss):.6f}  ref {float(ref_loss):.6f}  "
          f"grad norm {float(mpmd_norm):.4f}  ref {float(ref_norm):.4f}")
    np.testing.assert_allclose(float(mpmd_loss), float(ref_loss), rtol=1e-5)
    np.testing.assert_allclose(float(mpmd_norm), float(ref_norm), rtol=1e-5)
  for a, b in zip(jax.tree.leaves(jax.device_get(mpmd_params)),
                  jax.tree.leaves(jax.device_get(ref_params))):
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
  print(f"MPMD pipeline of MaxText {args.model} stages {stage_layers} ({args.schedule}) "
        "matches the single-mesh reference")
  return 0


if __name__ == "__main__":
  sys.exit(main())
