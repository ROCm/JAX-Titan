"""Pipeline-parallel train steps on Shardy MPMD.

Each pipeline stage is a named computation assigned to its own mesh. Each
microbatch is an `mpmd.call` with its own call counter, so Shardy's pipeline
schedules can interleave forward and backward fragments across microbatches.
"""

from collections.abc import Callable, Sequence

import jax
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


def make_train_step(
    stage_fns: Sequence[Callable[..., jax.Array]],
    optimizer: optax.GradientTransformation,
) -> Callable:
  """Returns `step(stage_params, opt_states, tokens, targets)`.

  Every stage keeps its own optimizer state, so gradient clipping is per stage.
  """

  def train_step(stage_params, opt_states, tokens, targets):
    loss, grads = jax.value_and_grad(
        lambda p: pipeline_loss(stage_fns, p, tokens, targets)
    )(stage_params)
    new_params, new_states = [], []
    for params, grad, state in zip(stage_params, grads, opt_states):
      updates, state = optimizer.update(grad, state, params)
      new_params.append(optax.apply_updates(params, updates))
      new_states.append(state)
    return tuple(new_params), tuple(new_states), loss

  return train_step
