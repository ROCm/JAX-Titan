import jax
import optax

from jaxtitan import parallel
from jaxtitan.models import llama

P = jax.sharding.PartitionSpec


def test_spec_layouts():
  mesh = {"data": 4, "tensor": 2}
  axes = ("layers", "embed", "heads")
  shape = (16, 4096, 4096)
  assert parallel.spec(axes, shape, {"data": 4, "tensor": 1}, fsdp=False) == P()
  assert parallel.spec(axes, shape, mesh, fsdp=False) == P(None, None, "tensor")
  assert parallel.spec(axes, shape, mesh, fsdp=True) == P(None, "data", "tensor")
  assert parallel.spec(("heads", "embed"), (4096, 4096), mesh, fsdp=True) == P(
      "tensor", "data"
  )


def test_spec_skips_layers_and_uneven_dims():
  mesh = {"data": 4, "tensor": 1}
  assert parallel.spec(("layers", "embed"), (16, 6), mesh, fsdp=True) == P()
  assert parallel.spec(("layers", "embed", "mlp"), (16, 6, 8), mesh, fsdp=True) == P(
      None, None, "data"
  )


def test_optimizer_state_follows_parameters():
  cfg = llama.FLAVORS["llama3-test"]
  params = jax.eval_shape(
      lambda k: llama.init_stage_params(cfg, k, stage=0, num_stages=2),
      jax.random.key(0),
  )
  state = jax.eval_shape(optax.adamw(1e-3).init, params)
  mesh = {"data": 2, "tensor": 2}
  param_specs = parallel.tree_specs(params, llama.logical_axes, mesh, fsdp=True)
  state_specs = parallel.tree_specs(state, llama.logical_axes, mesh, fsdp=True)
  assert param_specs["layers"]["wq"] == P(None, "data", "tensor")
  assert param_specs["layers"]["wo"] == P(None, "tensor", "data")
  assert param_specs["embed"] == P("data")
  adam = state_specs[0]
  assert adam.mu == param_specs and adam.nu == param_specs
  assert adam.count == P()


def test_without_axis():
  assert parallel.without_axis(P("data", "tensor"), "data", 3) == (None, "tensor", None)
  assert parallel.without_axis(P(("data", "tensor")), "data", 1) == ("tensor",)


def test_every_parameter_has_logical_axes():
  cfg = llama.FLAVORS["llama3-test"]
  params = jax.eval_shape(
      lambda k: llama.init_stage_params(cfg, k, stage=0, num_stages=1),
      jax.random.key(0),
  )
  for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
    assert llama.logical_axes(path, leaf) is not None, path
