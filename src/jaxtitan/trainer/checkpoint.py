"""Orbax checkpoints of the training state.

The state is any pytree of arrays. In an MPMD pipeline that is one subtree of
parameters and optimizer state per stage, each on its stage's mesh. Restoring
places every array on the sharding of the target state, so a checkpoint can
be restored into another sharding of the same stages, for example from data
parallelism to FSDP inside each stage.
"""

import pathlib

import jax
import orbax.checkpoint as ocp


def abstract_like(tree):
  """ShapeDtypeStructs with the shapes, dtypes and shardings of `tree`."""
  return jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), tree
  )


class Checkpointer:
  """Saves every `save_every` steps, keeps the latest `keep` checkpoints.

  Steps count completed training steps, so restoring step `k` resumes with
  the `k`-th step. `meta` is a JSON-serializable dict saved alongside, such as
  the data loader position.
  """

  def __init__(self, directory: str, *, save_every: int = 0, keep: int = 2):
    self._manager = ocp.CheckpointManager(
        pathlib.Path(directory).absolute(),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=save_every or 1,
            max_to_keep=keep,
            create=True,
        ),
        item_names=("state", "meta"),
    )
    self._save_every = save_every

  def latest_step(self) -> int | None:
    return self._manager.latest_step()

  def save(self, step: int, state, meta: dict, *, force: bool = False) -> bool:
    """Saves if `step` is on the interval, or if `force`; returns whether it did.

    Returns once the arrays are in host memory, so `state` may be donated
    right after; writing to storage continues in the background.
    """
    if not force and (not self._save_every or step % self._save_every):
      return False
    return self._manager.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state), meta=ocp.args.JsonSave(meta)
        ),
        force=True,
    )

  def restore(self, step: int, target):
    """Returns `(state, meta)`; `target` is a pytree of arrays or
    ShapeDtypeStructs giving the shapes, dtypes and shardings to restore to."""
    restored = self._manager.restore(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(target), meta=ocp.args.JsonRestore()
        ),
    )
    return restored.state, restored.meta

  def close(self):
    """Waits for pending saves to finish."""
    self._manager.wait_until_finished()
    self._manager.close()
