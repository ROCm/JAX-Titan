"""Data-parallel communication per training step, against the step's compute.

Every step, each GPU's gradient (its share of the parameters after tensor and pipeline
parallelism) is all-reduced over the data-parallel ranks; with FSDP the weights are
all-gathered in the forward and the backward pass and the gradient reduce-scattered
instead. Collectives run hierarchically: inside a scale-up domain over the GPU links,
then across domains, each GPU sending its 1/k share over its own NIC. A ring moves
(n - 1) / n of the data through each GPU for a reduce-scatter or an all-gather, twice
that for an all-reduce. Collectives reach `efficiency` of the links' peak: JAX-Titan's
RCCL Inspector measured 57-79% for 4-GPU all-reduces on MI355X. Overlapped with compute,
the communication is hidden while it takes less time than the step.
"""

import math

EFFICIENCY = 0.7


def data_parallel_seconds(params_per_gpu: float, dp: int, k: int, scaleup_gbs: float | None,
                          scaleout_gbs: float | None, *, grad_bytes: int = 4,
                          weight_bytes: int = 2, fsdp: bool = False,
                          efficiency: float = 1.0) -> dict:
  """Seconds per step of the data-parallel collectives; `k` of the `dp` ranks share a
  scale-up domain, whose links give them `scaleup_gbs` per direction."""
  n = math.ceil(dp / k)
  grads = params_per_gpu * grad_bytes
  # Halves of an all-reduce, or FSDP's two weight all-gathers and a gradient reduce-scatter.
  halves = [params_per_gpu * weight_bytes] * 2 + [grads] if fsdp else [grads, grads]
  inside = (sum(v * (k - 1) / k for v in halves) / (scaleup_gbs * 1e9 * efficiency)
            if k > 1 else 0.0)
  if n > 1 and not scaleout_gbs:
    raise ValueError("crosses scale-up domains without a scale-out bandwidth")
  across = (sum(v / k * (n - 1) / n for v in halves) / (scaleout_gbs * 1e9 * efficiency)
            if n > 1 else 0.0)
  return {"bytes_per_gpu": sum(halves), "ranks_per_domain": k, "domains": n,
          "efficiency": efficiency, "inside_s": inside, "across_s": across,
          "total_s": inside + across}


def ranks_per_domain(gpus: int, dp: int, tp: int, scaleup_gpus: int) -> int:
  """Data-parallel ranks inside one scale-up domain: tensor parallelism innermost, then
  data parallelism; pipeline stages share a domain only when the whole job fits one."""
  if gpus <= scaleup_gpus:
    return dp
  return max(1, min(dp, scaleup_gpus // tp))
