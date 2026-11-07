"""Pipeline-parallel train steps on Shardy MPMD.

Each pipeline stage is a named computation assigned to its own mesh. Each
microbatch is an `mpmd.call` with its own call counter, so Shardy's pipeline
schedules can interleave forward and backward fragments across microbatches.
"""

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from shardy.integrations.python.jax import mpmd
from shardy.integrations.python.jax.mpmd import pipeline_registry

from jaxtitan import parallel

DATA_AXIS = parallel.DATA_AXIS
TENSOR_AXIS = parallel.TENSOR_AXIS


def stage_name(i: int) -> str:
  return f"stage{i}"


def mesh_name(i: int) -> str:
  return f"mesh{i}"


def make_topology(
    num_stages: int,
    devices: Sequence[jax.Device] | None = None,
    *,
    tensor_parallel: int = 1,
) -> dict[str, jax.sharding.Mesh]:
  """Splits devices into one equally sized (data, tensor) mesh per stage.

  Consecutive devices of a stage form its tensor-parallel groups.
  """
  devices = list(jax.devices() if devices is None else devices)
  if len(devices) % num_stages:
    raise ValueError(f"{len(devices)} devices do not split into {num_stages} stages")
  per_stage = len(devices) // num_stages
  if per_stage % tensor_parallel:
    raise ValueError(
        f"{per_stage} devices per stage do not split into tensor-parallel "
        f"groups of {tensor_parallel}"
    )
  shape = (per_stage // tensor_parallel, tensor_parallel)
  return {
      mesh_name(i): jax.sharding.Mesh(
          np.asarray(devices[i * per_stage:(i + 1) * per_stage]).reshape(shape),
          (DATA_AXIS, TENSOR_AXIS),
      )
      for i in range(num_stages)
  }


def make_config(
    topology: dict[str, jax.sharding.Mesh],
    schedule: str | None = None,
    options: Mapping[str, bool | str] | None = None,
    *,
    num_stages: int | None = None,
) -> mpmd.Config:
  """`schedule` names an entry of Shardy's pipeline registry, e.g. "GPIPE".

  `options` are extra Shardy MPMD partitioning options. With a schedule,
  `mpmd_merge_inferred_after_scheduling` defaults to True: merging inferred
  fragments first folds the optimizer update into one microbatch's backward
  fragment, which then has to run after every other backward fragment.

  With more stages than meshes, as circular schedules need, stage `i` runs on
  mesh `i % len(topology)`.

  Input and output meshes are taken from the `jit` shardings rather than
  inferred, since inference may move per-stage state such as optimizer step
  counters to another mesh.
  """
  n = len(topology)
  num_stages = num_stages or n
  if num_stages % n:
    raise ValueError(f"{num_stages} stages do not split evenly over {n} meshes")
  if schedule:
    options = {"mpmd_merge_inferred_after_scheduling": True, **(options or {})}
  return mpmd.make_config(
      topology,
      {stage_name(i): mesh_name(i % n) for i in range(num_stages)},
      name_to_stage_assignment={stage_name(i): i for i in range(num_stages)},
      partitioning_options=dict(options) if options else None,
      read_input_output_mesh_from_shardings=True,
      pipeline_schedule=(
          pipeline_registry.get_pipeline_schedule(schedule) if schedule else None
      ),
  )


def pipeline_loss(
    stage_fns: Sequence[Callable[..., jax.Array]],
    stage_params: Sequence,
    tokens,
    targets,
) -> jax.Array:
  """Mean loss over microbatches of [microbatches, batch, seq] inputs.

  `tokens` and `targets` may also be pytrees of arrays that each lead with the
  microbatch axis. Stage `i` is called as `stage_fns[i](params, x)`; the last
  stage also takes the targets and returns the microbatch loss.
  """
  last = len(stage_fns) - 1

  def one_microbatch(stage_params, tok, tgt):
    x = tok
    for i, fn in enumerate(stage_fns):
      args = (stage_params[i], x, tgt) if i == last else (stage_params[i], x)
      x = mpmd.named_computation(fn, name=stage_name(i))(*args)
    return x

  num_microbatches = jax.tree.leaves(tokens)[0].shape[0]
  total = 0.0
  for m in range(num_microbatches):
    total = total + mpmd.call(one_microbatch, call_counter=m)(
        stage_params, jax.tree.map(lambda a, m=m: a[m], tokens),
        jax.tree.map(lambda a, m=m: a[m], targets),
    )
  return total / num_microbatches


def _device_copies(params, axes, specs, mesh: jax.sharding.Mesh):
  """Stacks `params` once per data-parallel rank of `mesh` along `axes`.

  Each copy keeps the tensor-parallel sharding of its parameter. Parameters
  sharded over the data axis are all-gathered into their copies.
  """
  n = mesh.shape[DATA_AXIS]

  def copy(x, axis, s):
    entries = parallel.without_axis(s, DATA_AXIS, x.ndim)
    shape = x.shape[:axis] + (n,) + x.shape[axis:]
    copy_spec = jax.sharding.PartitionSpec(*entries[:axis], DATA_AXIS, *entries[axis:])
    return jax.lax.with_sharding_constraint(
        jnp.broadcast_to(jnp.expand_dims(x, axis), shape),
        jax.sharding.NamedSharding(mesh, copy_spec),
    )

  return jax.tree.map(copy, params, axes, specs)


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


def _per_device_manual(fn: Callable[..., jax.Array], axes, mesh: jax.sharding.Mesh, last: bool):
  """`_per_device` as a shard_map over the data axis.

  Each device calls `fn` once on its own copy and share, without a batch
  axis, so ops with no batching rule (such as custom_partitioning kernels)
  run. Axes of size 1 are manual too, so without tensor parallelism the
  shard_map is fully manual: custom_partitioning kernels cannot be
  partitioned inside a partly manual one.
  """
  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  manual = {DATA_AXIS} | {a for a in mesh.axis_names if mesh.shape[a] == 1}

  def local(copies, x, *targets):
    params = jax.tree.map(jnp.squeeze, copies, axes)
    out = fn(params, x, *targets)
    return jax.lax.pmean(out, DATA_AXIS) if last else out

  def per_device_fn(copies, x, *targets):
    copy_specs = jax.tree.map(lambda _, axis: P(*([None] * axis), DATA_AXIS), copies, axes)
    return jax.shard_map(
        local, mesh=mesh, axis_names=manual, check_vma=False,
        in_specs=(copy_specs, P(DATA_AXIS)) + (P(DATA_AXIS),) * len(targets),
        out_specs=P() if last else P(DATA_AXIS),
    )(copies, x, *targets)

  return per_device_fn


def global_grad_norm(stage_grads: Sequence) -> jax.Array:
  """The norm of all stages' gradients together, usable on every stage's mesh.

  Each stage sums its squares on its own mesh; Shardy MPMD moves those scalars
  to one mesh for the sum and the result back to every mesh. Under `jax.jit`
  the MPMD annotations do nothing.
  """
  total = None
  for grads in stage_grads:
    squares = sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in jax.tree.leaves(grads))
    total = squares if total is None else total + squares
  return mpmd.broadcast(mpmd.reduce(jnp.sqrt(total)))


def clipped_update(optimizer: optax.GradientTransformation, params, grads, state,
                   norm: jax.Array, max_grad_norm: float | None = None):
  """`optimizer`'s update of one stage, given the global gradient norm.

  With `max_grad_norm`, `grads` are first scaled as
  `optax.clip_by_global_norm(max_grad_norm)` scales them with `norm` as the
  global norm. When `norm` is not finite, `params` and `state` come back
  unchanged.
  """
  if max_grad_norm is not None:
    clip = norm < max_grad_norm
    grads = jax.tree.map(
        lambda g: jax.lax.select(clip, g, (g / norm.astype(g.dtype)) * max_grad_norm), grads)
  updates, new_state = optimizer.update(grads, state, params)
  new_params = optax.apply_updates(params, updates)
  finite = jnp.isfinite(norm)

  def keep(new, old):
    return jnp.where(finite, new, old)

  return jax.tree.map(keep, new_params, params), jax.tree.map(keep, new_state, state)


def make_train_step(
    stage_fns: Sequence[Callable[..., jax.Array]],
    optimizer: optax.GradientTransformation,
    *,
    max_grad_norm: float | None = None,
    copy_axes: Sequence | None = None,
    param_specs: Sequence | None = None,
    mesh: jax.sharding.Mesh | None = None,
    per_device: str = "vmap",
) -> Callable:
  """Returns `step(stage_params, opt_states, tokens, targets)`.

  The step returns `(stage_params, opt_states, loss, grad_norm)`. Every stage
  keeps its own optimizer state, but `grad_norm` is one norm over all stages'
  gradients, and `max_grad_norm` clips by it, as for a single model. When the
  norm is not finite, no stage updates and the loss comes back as NaN.

  By default a stage reduces its gradients across its data-parallel ranks in
  the backward pass of every microbatch. With `copy_axes`, one pytree of axes
  per stage, each rank trains its own copy of the stage parameters stacked
  along those axes: microbatch gradients stay on their rank, and the stage
  sums them across ranks once per step. `stage_fns` then see one rank's share
  of a microbatch and must not constrain shardings.

  `param_specs`, one pytree of PartitionSpecs per stage, gives the sharding of
  each stage's parameters inside its mesh; the default is replicated. Copies
  keep the tensor-parallel part of it. Summed gradients take the full spec, so
  parameters sharded over the data axis get their gradients reduce-scattered.

  `per_device` picks how each rank's copy runs: `vmap` over the copies, or
  `shard_map` over the data axis, which also runs ops that have no batching
  rule.

  `mesh` is any one stage mesh. `jax.jit` requires every sharding constraint
  in a program to name the same devices; Shardy MPMD only uses its axes.
  """
  if per_device not in ("vmap", "shard_map"):
    raise ValueError(f"per_device={per_device!r}; expected vmap or shard_map")
  if copy_axes is not None:
    last = len(stage_fns) - 1
    n = mesh.shape[DATA_AXIS]
    if param_specs is None:
      param_specs = [
          jax.tree.map(lambda _: jax.sharding.PartitionSpec(), axes)
          for axes in copy_axes
      ]
    if per_device == "vmap":
      device_fns = [
          _per_device(fn, copy_axes[i], n, i == last) for i, fn in enumerate(stage_fns)
      ]
    else:
      device_fns = [
          _per_device_manual(fn, copy_axes[i], mesh, i == last)
          for i, fn in enumerate(stage_fns)
      ]

  def loss_and_grads(stage_params, tokens, targets):
    if copy_axes is None:
      return jax.value_and_grad(
          lambda p: pipeline_loss(stage_fns, p, tokens, targets)
      )(stage_params)
    copies = [
        _device_copies(p, copy_axes[i], param_specs[i], mesh)
        for i, p in enumerate(stage_params)
    ]
    loss, copy_grads = jax.value_and_grad(
        lambda c: pipeline_loss(device_fns, c, tokens, targets)
    )(copies)

    def reduce(g, axis, s):
      return jax.lax.with_sharding_constraint(
          g.sum(axis), jax.sharding.NamedSharding(mesh, s)
      )

    grads = [
        jax.tree.map(reduce, g, copy_axes[i], param_specs[i])
        for i, g in enumerate(copy_grads)
    ]
    return loss, grads

  def train_step(stage_params, opt_states, tokens, targets):
    loss, grads = loss_and_grads(stage_params, tokens, targets)
    norm = global_grad_norm(grads)
    new_params, new_states = [], []
    for params, grad, state in zip(stage_params, grads, opt_states):
      params, state = clipped_update(optimizer, params, grad, state, norm, max_grad_norm)
      new_params.append(params)
      new_states.append(state)
    return tuple(new_params), tuple(new_states), jnp.where(jnp.isfinite(norm), loss, jnp.nan), norm

  return train_step
