"""Kernel calls as custom_partitioning ops that shard by named factors.

Providers whose kernels have no sharding rules of their own (jax-triton's
`triton_call`) describe each operand and result with factor names, as in an
einsum; the kernel then runs on local shards.
"""

import functools

import jax
from jax.experimental.custom_partitioning import custom_partitioning


def partitioned(local, in_dims, out_dims, replicated=()):
  """`local(*arrays, params)` as an op that shards by factor.

  `in_dims` and `out_dims` name every dimension of each operand and result with
  a factor. A factor shards like the first operand dimension that names it,
  unless its mesh axes already shard another factor, and except the
  `replicated` factors, which the kernel needs whole. A result without a
  sharded factor is a sum over it, so it is summed across that factor's
  shards. `params` holds the static arguments and must be hashable.
  """
  n_in = len(in_dims)
  multiple = len(out_dims) > 1

  @functools.partial(custom_partitioning, static_argnums=(n_in,))
  def op(*args):
    return local(*args)

  def factor_axes(arg_shapes):
    axes, used = {}, set()
    for dims, arg in zip(in_dims, arg_shapes):
      spec = tuple(arg.sharding.spec) + (None,) * (len(dims) - len(arg.sharding.spec))
      for factor, entry in zip(dims, spec):
        if factor in replicated or factor in axes:
          continue
        names = () if entry is None else (entry,) if isinstance(entry, str) else tuple(entry)
        if used.intersection(names):
          names = ()
        used.update(names)
        axes[factor] = names or None
    return axes

  def shardings(mesh, dims_list, axes):
    return tuple(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(
        *(axes.get(f) for f in dims))) for dims in dims_list)

  def results(outs):
    return outs if multiple else outs[0]

  def summed_over(dims, axes):
    return tuple(a for f, names in axes.items() if f not in dims and names for a in names)

  def partition(params, mesh, arg_shapes, result_shapes):
    del result_shapes
    axes = factor_axes(arg_shapes)

    def lowered(*args):
      outs = local(*args, params)
      outs = outs if multiple else (outs,)
      return results(tuple(jax.lax.psum(o, names) if (names := summed_over(dims, axes)) else o
                           for dims, o in zip(out_dims, outs)))

    return (mesh, lowered, results(shardings(mesh, out_dims, axes)),
            shardings(mesh, in_dims, axes))

  def infer(params, mesh, arg_shapes, result_shapes):
    del params, result_shapes
    return results(shardings(mesh, out_dims, factor_axes(arg_shapes)))

  rule = " -> ".join(", ".join(" ".join(d) for d in dims) for dims in (in_dims, out_dims))
  op.def_partition(partition, infer_sharding_from_operands=infer, sharding_rule=rule,
                   need_replication_factors=tuple(replicated))
  return op
