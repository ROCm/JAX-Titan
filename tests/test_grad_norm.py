"""The global gradient norm across stages: clipping and non-finite steps."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from shardy.integrations.python.jax import mpmd

from jaxtitan import mpmd as jt_mpmd


def test_clipped_update_matches_optax_clip():
  rng = np.random.default_rng(0)
  params = {"a": jnp.asarray(rng.standard_normal((3, 4)), jnp.float32), "b": jnp.ones(5)}
  grads = jax.tree.map(lambda p: 3.0 * jnp.ones_like(p), params)
  opt = optax.adamw(0.1)
  reference = optax.chain(optax.clip_by_global_norm(1.0), opt)
  ref_updates, _ = reference.update(grads, reference.init(params), params)
  norm = jt_mpmd.global_grad_norm([grads])
  new, _ = jt_mpmd.clipped_update(opt, params, grads, opt.init(params), norm, 1.0)
  np.testing.assert_allclose(float(norm), float(optax.tree.norm(grads)), rtol=1e-6)
  jax.tree.map(lambda n, p, u: np.testing.assert_allclose(n, p + u, rtol=1e-6),
               new, params, ref_updates)


def test_pipeline_step_skips_non_finite_gradients():
  topology = jt_mpmd.make_topology(2, jax.devices()[:8])
  meshes = list(topology.values())
  config = jt_mpmd.make_config(topology, "ONE_FWD_ONE_BWD")
  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  def stage0(p, x):
    # sqrt'(0) is infinite: the loss stays finite and the gradient does not.
    return x * p["w"] + jnp.sqrt(jnp.abs(p["v"]))

  def stage1(p, y, t):
    return jnp.mean((y * p["w"] - t) ** 2)

  params = ({"w": jnp.ones(4), "v": jnp.zeros(4)}, {"w": jnp.ones(4)})
  opt = optax.sgd(0.1)
  states = tuple(opt.init(p) for p in params)

  def placed(tree, mesh, spec=P()):
    return jax.tree.map(lambda _: jax.sharding.NamedSharding(mesh, spec), tree)

  p_sh = tuple(placed(p, m) for p, m in zip(params, meshes))
  s_sh = tuple(placed(s, m) for s, m in zip(states, meshes))
  x_sh = jax.sharding.NamedSharding(meshes[0], P(None, jt_mpmd.DATA_AXIS))
  t_sh = jax.sharding.NamedSharding(meshes[1], P(None, jt_mpmd.DATA_AXIS))
  scalar = jax.sharding.NamedSharding(meshes[1], P())
  xs = jnp.ones((2, 8, 4))
  step = jt_mpmd.make_train_step([stage0, stage1], opt, max_grad_norm=1.0)
  compiled = mpmd.jit(step, config, in_shardings=(p_sh, s_sh, x_sh, t_sh),
                      out_shardings=(p_sh, s_sh, scalar, scalar)).lower(
                          params, states, xs, xs).compile()
  new, _, loss, norm = compiled(jax.device_put(params, p_sh), jax.device_put(states, s_sh),
                                jax.device_put(xs, x_sh), jax.device_put(xs, t_sh))
  assert not np.isfinite(float(norm))
  assert np.isnan(float(loss))
  jax.tree.map(np.testing.assert_array_equal, jax.device_get(new), jax.device_get(params))
