"""Memory per GPU of a training layout: state, activations and logits.

An analytic floor, in bytes, for the busiest GPU (the first pipeline stage holds the most
microbatches in flight; the last also holds the output head and logits):

- state: parameters, gradients and optimizer state per parameter (by default fp32
  weights, fp32 gradients and fp32 AdamW moments: 4 + 4 + 8 bytes), divided over tensor
  and pipeline parallelism, and over the data-parallel ranks with FSDP; routed experts
  divided over expert parallelism instead. `grad_copies` adds a per-rank copy of the
  weights and gradients, which JAX-Titan's per-step gradient reduction keeps.
- activations per token per layer stored for the backward pass, in bf16: the layer
  input, normalized inputs, Q, K, V, the attention output and the MLP's two inputs and
  activation output, `6e + 2*kv*hd + 3f` elements (flash attention keeps no scores);
  with rematerialization, only the layer input plus one layer recomputed at a time.
  Divided by tensor parallelism with sequence parallelism.
- logits: fp32 logits and their gradient on the last stage.

XLA adds temporaries, unaliased update buffers and fragmentation on top; `calibrate`
measures the ratio of a recorded run's peak to its floor.
"""

import dataclasses
import pathlib

from jax_titan.planning import models
from jax_titan.planning.models import Model

ACT_BYTES = 2


@dataclasses.dataclass(frozen=True)
class Layout:
  data_parallel: int = 1
  tensor_parallel: int = 1
  pipeline_stages: int = 1
  expert_parallel: int = 1
  fsdp: bool = False
  microbatch_size: int = 1  # sequences per microbatch, over the data-parallel ranks
  microbatches: int = 1
  seq_len: int = 2048
  remat: bool = True
  schedule: str = "1f1b"  # "1f1b" (and zero-bubble): up to `pipeline_stages` in flight
  weight_bytes: int = 4
  grad_bytes: int = 4
  optimizer_bytes: int = 8
  grad_copies: bool = False

  @property
  def gpus(self) -> int:
    return self.data_parallel * self.tensor_parallel * self.pipeline_stages


def activation_elements_per_token_layer(model: Model) -> float:
  """Mean over the layers: MoE layers store the active experts' MLP activations."""
  e, kv = model.emb, model.kv_heads * model.head_dim
  moe_f = (model.experts_per_token + model.shared_experts) * model.expert_mlp
  f = (model.dense_layers * model.mlp + model.moe_layers * moe_f) / model.layers
  return 6 * e + 2 * kv + 3 * f


def per_gpu(model: Model, layout: Layout) -> dict:
  lo = layout
  stage_layers = model.layers / lo.pipeline_stages
  dense_params = model.params() - model.expert_param_count()
  per_stage_dense = (dense_params - model.embedding_params()) / lo.pipeline_stages
  per_stage_experts = model.expert_param_count() / lo.pipeline_stages
  # The busiest stage: the last holds the output head and the first the embedding, one
  # stage both.
  ends = model.embedding_params() if lo.pipeline_stages == 1 else model.vocab * model.emb
  params = (per_stage_dense + ends) / lo.tensor_parallel
  params += per_stage_experts / (lo.expert_parallel * lo.tensor_parallel)
  sharded = params / lo.data_parallel if lo.fsdp else params
  per_param = lo.weight_bytes + lo.grad_bytes + lo.optimizer_bytes
  state = sharded * per_param
  if lo.grad_copies:
    state += params * (lo.weight_bytes + lo.grad_bytes)

  local_tokens = lo.microbatch_size / lo.data_parallel * lo.seq_len
  full_layer = local_tokens * activation_elements_per_token_layer(model) * ACT_BYTES
  full_layer /= lo.tensor_parallel
  if lo.remat:
    per_microbatch = stage_layers * local_tokens * model.emb * ACT_BYTES + full_layer
  else:
    per_microbatch = stage_layers * full_layer
  in_flight = lo.microbatches if lo.schedule == "gpipe" else min(lo.pipeline_stages,
                                                                 lo.microbatches)
  activations = in_flight * per_microbatch
  logits = 2 * local_tokens * model.vocab * 4 / lo.tensor_parallel
  total = state + activations + logits
  return {"params_per_gpu": params, "state_bytes": state, "activation_bytes": activations,
          "logits_bytes": logits, "total_bytes": total, "in_flight_microbatches": in_flight}


def run_layout(args: dict, num_devices: int) -> Layout:
  """The layout of a `jax_titan.trainer.train` run, from its start event's arguments."""
  mpmd = args.get("parallel", "mpmd") == "mpmd"
  stages = args.get("num_stages", 1) if mpmd else 1
  tp = args.get("tensor_parallel", 1)
  reduce_grads = args.get("reduce_grads") or ("microbatch" if ":" in args["model"] else "step")
  microbatches, size = args.get("microbatches", 8), args.get("microbatch_size", 4)
  if not mpmd:  # dp and fsdp run all microbatches as one batch
    microbatches, size = 1, microbatches * size
  return Layout(
      data_parallel=num_devices // (stages * tp), tensor_parallel=tp, pipeline_stages=stages,
      fsdp=bool(args.get("stage_fsdp")) or args.get("parallel") == "fsdp",
      microbatch_size=size, microbatches=microbatches, seq_len=args.get("seq_len", 2048),
      remat=args.get("remat", True),
      schedule="gpipe" if str(args.get("schedule")).upper() == "GPIPE" else "1f1b",
      grad_copies=mpmd and stages > 1 and reduce_grads == "step")


def calibrate(run) -> dict:
  """A recorded run's peak HBM over the floor of its layout; `run` is a --run_dir or its
  train.jsonl."""
  from jax_titan.observability import records  # pylint: disable=g-import-not-at-top

  path = pathlib.Path(run)
  log = records.read(path / "train.jsonl" if path.is_dir() else path)
  starts = records.events(log, "start")
  peaks = [g for r in records.events(log, "memory") for g in r["peak_hbm_gib"]]
  if not starts or not peaks:
    raise ValueError(f"{run} has no start or memory event")
  start = starts[0]
  model = models.from_name(start["args"]["model"])
  layout = run_layout(start["args"], start["num_devices"])
  floor = per_gpu(model, layout)["total_bytes"]
  peak = max(peaks) * 2**30
  return {"model": model.name, "layout": dataclasses.asdict(layout), "floor_bytes": floor,
          "peak_bytes": peak, "overhead": peak / floor}
