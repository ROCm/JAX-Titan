"""Shardings of model state inside one SPMD mesh: DP, FSDP and TP.

A model names the logical axes of each parameter, such as `layers`, `embed`
or `heads`. Tensor parallelism shards one axis from `TENSOR_PARALLEL_AXES` over
the mesh's tensor axis. FSDP then shards the first other dimension that splits
evenly over the data axis, never `layers`, which `lax.scan` iterates over.
Without either, state is replicated (data parallelism).
"""

from collections.abc import Callable, Mapping, Sequence

import jax

DATA_AXIS = "data"
TENSOR_AXIS = "tensor"
TENSOR_PARALLEL_AXES = frozenset({"heads", "kv_heads", "mlp", "vocab"})

P = jax.sharding.PartitionSpec
LogicalAxes = Callable[[tuple, jax.ShapeDtypeStruct], Sequence[str | None] | None]


def spec(
    axes: Sequence[str | None],
    shape: Sequence[int],
    mesh_shape: Mapping[str, int],
    *,
    fsdp: bool,
) -> jax.sharding.PartitionSpec:
  """The PartitionSpec of one array with logical `axes`."""
  out = [None] * len(shape)
  tp = mesh_shape.get(TENSOR_AXIS, 1)
  if tp > 1:
    for dim, axis in enumerate(axes):
      if axis in TENSOR_PARALLEL_AXES and shape[dim] % tp == 0:
        out[dim] = TENSOR_AXIS
        break
  dp = mesh_shape.get(DATA_AXIS, 1)
  if fsdp and dp > 1:
    for dim, axis in enumerate(axes):
      if axis != "layers" and out[dim] is None and shape[dim] % dp == 0:
        out[dim] = DATA_AXIS
        break
  # MPMD executables return shardings without trailing Nones, and a train
  # step's inputs must compare equal to its previous outputs.
  while out and out[-1] is None:
    out.pop()
  return P(*out)


def tree_specs(tree, logical_axes: LogicalAxes, mesh_shape, *, fsdp: bool):
  """PartitionSpecs for parameters, or optimizer state that mirrors them.

  `logical_axes(path, leaf)` names the axes of a leaf; leaves it returns None
  for, such as optimizer step counts, are replicated.
  """

  def leaf_spec(path, leaf):
    axes = logical_axes(path, leaf)
    return P() if axes is None else spec(axes, leaf.shape, mesh_shape, fsdp=fsdp)

  return jax.tree_util.tree_map_with_path(leaf_spec, tree)


def without_axis(
    s: jax.sharding.PartitionSpec, axis: str, ndim: int
) -> tuple[str | tuple[str, ...] | None, ...]:
  """The entries of `s` for `ndim` dimensions, with mesh axis `axis` removed."""

  def drop(entry):
    if entry == axis:
      return None
    if isinstance(entry, tuple):
      kept = tuple(a for a in entry if a != axis)
      return kept if len(kept) > 1 else (kept[0] if kept else None)
    return entry

  entries = tuple(s) + (None,) * (ndim - len(s))
  return tuple(drop(e) for e in entries)
