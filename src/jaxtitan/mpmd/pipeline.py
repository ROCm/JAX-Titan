"""Pipeline-parallel train steps on Shardy MPMD.

Each pipeline stage is a named computation assigned to its own mesh. Each
microbatch is an `mpmd.call` with its own call counter, so Shardy's pipeline
schedules can interleave forward and backward fragments across microbatches.
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from shardy.integrations.python.jax import mpmd
from shardy.integrations.python.jax.mpmd import pipeline_registry

DATA_AXIS = "data"


def stage_name(i: int) -> str:
  return f"stage{i}"


def mesh_name(i: int) -> str:
  return f"mesh{i}"


def make_topology(
    num_stages: int, devices: Sequence[jax.Device] | None = None
) -> dict[str, jax.sharding.Mesh]:
  """Splits devices into one equally sized data-parallel mesh per stage."""
  devices = list(jax.devices() if devices is None else devices)
  if len(devices) % num_stages:
    raise ValueError(f"{len(devices)} devices do not split into {num_stages} stages")
  per_stage = len(devices) // num_stages
  return {
      mesh_name(i): jax.sharding.Mesh(
          np.asarray(devices[i * per_stage:(i + 1) * per_stage]), (DATA_AXIS,)
      )
      for i in range(num_stages)
  }


def make_config(
    topology: dict[str, jax.sharding.Mesh], schedule: str | None = None
) -> mpmd.Config:
  """`schedule` names an entry of Shardy's pipeline registry, e.g. "GPIPE".

  Input and output meshes are taken from the `jit` shardings rather than
  inferred, since inference may move per-stage state such as optimizer step
  counters to another mesh.
  """
  n = len(topology)
  return mpmd.make_config(
      topology,
      {stage_name(i): mesh_name(i) for i in range(n)},
      name_to_stage_assignment={stage_name(i): i for i in range(n)},
      read_input_output_mesh_from_shardings=True,
      pipeline_schedule=(
          pipeline_registry.get_pipeline_schedule(schedule) if schedule else None
      ),
  )


def pipeline_loss(
    stage_fns: Sequence[Callable[..., jax.Array]],
    stage_params: Sequence,
    tokens: jax.Array,
    targets: jax.Array,
) -> jax.Array:
  """Mean loss over microbatches of [microbatches, batch, seq] inputs.

  Stage `i` is called as `stage_fns[i](params, x)`; the last stage also takes
  the targets and returns the microbatch loss.
  """
  last = len(stage_fns) - 1

  def one_microbatch(stage_params, tok, tgt):
    x = tok
    for i, fn in enumerate(stage_fns):
      args = (stage_params[i], x, tgt) if i == last else (stage_params[i], x)
      x = mpmd.named_computation(fn, name=stage_name(i))(*args)
    return x

  num_microbatches = tokens.shape[0]
  total = 0.0
  for m in range(num_microbatches):
    total = total + mpmd.call(one_microbatch, call_counter=m)(
        stage_params, tokens[m], targets[m]
    )
  return total / num_microbatches


def _device_copies(params, axes, mesh: jax.sharding.Mesh):
  """Stacks `params` once per device of `mesh` along `axes`, one copy each."""
  n = mesh.shape[DATA_AXIS]

  def copy(x, axis):
    shape = x.shape[:axis] + (n,) + x.shape[axis:]
    spec = jax.sharding.PartitionSpec(*([None] * axis + [DATA_AXIS]))
    return jax.lax.with_sharding_constraint(
        jnp.broadcast_to(jnp.expand_dims(x, axis), shape),
        jax.sharding.NamedSharding(mesh, spec),
    )

  return jax.tree.map(copy, params, axes)


def _per_device(fn: Callable[..., jax.Array], axes, n: int, last: bool):
  """Runs `fn` on each device's parameter copy and share of the microbatch."""

  def split(a):
    return a.reshape(n, a.shape[0] // n, *a.shape[1:])

  def per_device_fn(copies, x, *targets):
    out = jax.vmap(fn, in_axes=(axes, 0) + (0,) * len(targets))(
        copies, split(x), *map(split, targets)
    )
    return out.mean() if last else out.reshape(-1, *out.shape[2:])

  return per_device_fn


def make_train_step(
    stage_fns: Sequence[Callable[..., jax.Array]],
    optimizer: optax.GradientTransformation,
    *,
    copy_axes: Sequence | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> Callable:
  """Returns `step(stage_params, opt_states, tokens, targets)`.

  Every stage keeps its own optimizer state, so gradient clipping is per stage.

  By default a stage all-reduces its gradients across its devices in the
  backward pass of every microbatch. With `copy_axes`, one pytree of axes per
  stage, each device trains its own copy of the stage parameters stacked
  along those axes: microbatch gradients stay on their device, and the stage
  sums them across devices once per step. `stage_fns` then see one device's
  share of a microbatch and must not constrain shardings.

  `mesh` is any one stage mesh. `jax.jit` requires every sharding constraint
  in a program to name the same devices; Shardy MPMD only uses its axes.
  """
  if copy_axes is not None:
    last = len(stage_fns) - 1
    n = mesh.shape[DATA_AXIS]
    device_fns = [
        _per_device(fn, copy_axes[i], n, i == last) for i, fn in enumerate(stage_fns)
    ]

  def loss_and_grads(stage_params, tokens, targets):
    if copy_axes is None:
      return jax.value_and_grad(
          lambda p: pipeline_loss(stage_fns, p, tokens, targets)
      )(stage_params)
    copies = [
        _device_copies(p, copy_axes[i], mesh) for i, p in enumerate(stage_params)
    ]
    loss, copy_grads = jax.value_and_grad(
        lambda c: pipeline_loss(device_fns, c, tokens, targets)
    )(copies)
    grads = [
        jax.tree.map(lambda g, axis: g.sum(axis), g, copy_axes[i])
        for i, g in enumerate(copy_grads)
    ]
    return loss, grads

  def train_step(stage_params, opt_states, tokens, targets):
    loss, grads = loss_and_grads(stage_params, tokens, targets)
    new_params, new_states = [], []
    for params, grad, state in zip(stage_params, grads, opt_states):
      updates, state = optimizer.update(grad, state, params)
      new_params.append(optax.apply_updates(params, updates))
      new_states.append(state)
    return tuple(new_params), tuple(new_states), loss

  return train_step
