"""Where a layout's data-parallel ranks sit, for its collectives (`planning.collectives`)."""


def ranks_per_domain(gpus: int, dp: int, tp: int, scaleup_gpus: int) -> int:
  """Data-parallel ranks inside one scale-up domain: tensor parallelism innermost, then
  data parallelism; pipeline stages share a domain only when the whole job fits one."""
  if gpus <= scaleup_gpus:
    return dp
  return max(1, min(dp, scaleup_gpus // tp))
