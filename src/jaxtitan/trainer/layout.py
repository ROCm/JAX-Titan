"""What a trainer builder returns: how training state is placed and stepped,
and what one step processes."""

import dataclasses
from typing import Any, Callable


@dataclasses.dataclass
class Layout:
  """How the training state is placed and stepped.

  `abstract_state` has the tree structure, shapes, dtypes and shardings of
  `init()`'s `(params, opt_state)`, so a checkpoint can be restored without
  initializing first. `put_batch` places a host batch as the step's inputs
  and targets, which `input_spec` and `target_spec` describe.
  """

  init: Callable[[], tuple]
  abstract_state: tuple
  step: Any
  input_spec: Any
  target_spec: Any
  put_batch: Callable
  structure: str


@dataclasses.dataclass
class Workload:
  """What one training step processes.

  Token models train on token windows below `vocab_size`. Models without a
  dataset yet give `next_batch(step)`, a synthetic host batch for `put_batch`.
  """

  num_params: int
  tokens_per_step: int
  flops_per_step: float
  vocab_size: int | None = None
  next_batch: Callable[[int], Any] | None = None


def abstract(shapes, shardings):
  """ShapeDtypeStructs of `shapes` with `shardings`."""
  import jax  # pylint: disable=g-import-not-at-top

  return jax.tree.map(
      lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), shapes, shardings
  )
