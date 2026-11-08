"""Shardings of model state inside one SPMD mesh: DP, FSDP and TP."""

from jax_titan.parallel.sharding import DATA_AXIS
from jax_titan.parallel.sharding import TENSOR_AXIS
from jax_titan.parallel.sharding import TENSOR_PARALLEL_AXES
from jax_titan.parallel.sharding import spec
from jax_titan.parallel.sharding import tree_specs
from jax_titan.parallel.sharding import without_axis

__all__ = [
    "DATA_AXIS",
    "TENSOR_AXIS",
    "TENSOR_PARALLEL_AXES",
    "spec",
    "tree_specs",
    "without_axis",
]
