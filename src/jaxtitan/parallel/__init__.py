"""Shardings of model state inside one SPMD mesh: DP, FSDP and TP."""

from jaxtitan.parallel.sharding import DATA_AXIS
from jaxtitan.parallel.sharding import TENSOR_AXIS
from jaxtitan.parallel.sharding import TENSOR_PARALLEL_AXES
from jaxtitan.parallel.sharding import spec
from jaxtitan.parallel.sharding import tree_specs
from jaxtitan.parallel.sharding import without_axis

__all__ = [
    "DATA_AXIS",
    "TENSOR_AXIS",
    "TENSOR_PARALLEL_AXES",
    "spec",
    "tree_specs",
    "without_axis",
]
