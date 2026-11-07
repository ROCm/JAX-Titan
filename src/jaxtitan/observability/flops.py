"""Training FLOPs and communication of a Llama-family step, per op and per stage.

  python -m jaxtitan.observability.flops --model=llama3.1-8b --num_stages=2
      [--stage_layers=17,15] [--seq_len=2048] [--microbatches=8] [--microbatch_size=4]
      [--devices=8] [--tensor_parallel=1] [--reduce_grads=step] [--stage_fsdp] [--no-remat]
      [--peak_tflops=1436] [--busbw_gbs=160] [--json OUT]

`--model` is a JAX-Titan Llama flavor or `maxtext:<name>` (with MaxText installed).

Model FLOPs follow the usual convention: a matmul's backward pass costs twice its forward
(activation and weight gradients), so training costs three forward passes, and causal
attention counts half of its scores. MFU divides these. Executed FLOPs add what the
implementation recomputes: with rematerialized layers, every decoder layer's forward pass
runs again during the backward pass; HFU divides these. Elementwise ops (norms, RoPE, SiLU,
softmax, the loss) are left out.

Per stage it gives FLOPs, the time they take at `--peak_tflops` per GPU, and the layer
split that balances executed FLOPs across stages (the last stage also holds the output
head). Communication is the bytes each GPU sends per step: the gradient reduction inside a
stage (a ring all-reduce, or with `--stage_fsdp` an all-gather of the weights and a
reduce-scatter of the gradients), tensor-parallel all-reduces of activations, and the
activations and gradients passed between stages. `--busbw_gbs` turns bytes into time.
"""

import argparse
import dataclasses
import json

PARAM_BYTES = 4  # fp32 weights and gradients
ACT_BYTES = 2  # bf16 activations


@dataclasses.dataclass(frozen=True)
class Dims:
  vocab: int
  emb: int
  layers: int
  q_heads: int
  kv_heads: int
  head_dim: int
  mlp: int

  @classmethod
  def of(cls, cfg) -> "Dims":
    """From a `jaxtitan.models.llama.LlamaConfig`."""
    return cls(cfg.vocab_size, cfg.emb_dim, cfg.num_layers, cfg.num_query_heads,
               cfg.num_kv_heads, cfg.head_dim, cfg.mlp_dim)


def layer_forward(d: Dims, seq_len: int) -> dict[str, float]:
  """Forward FLOPs per token of one decoder layer, per op."""
  e, h, k, hd, f = d.emb, d.q_heads, d.kv_heads, d.head_dim, d.mlp
  return {
      "q_proj": 2 * e * h * hd,
      "k_proj": 2 * e * k * hd,
      "v_proj": 2 * e * k * hd,
      # QK^T and AV over the causal half of seq_len keys.
      "attention": 2 * h * hd * seq_len,
      "o_proj": 2 * h * hd * e,
      "mlp_gate_up": 4 * e * f,
      "mlp_down": 2 * f * e,
  }


def head_forward(d: Dims) -> float:
  return 2 * d.emb * d.vocab


def layer_params(d: Dims) -> int:
  return d.emb * d.head_dim * (2 * d.q_heads + 2 * d.kv_heads) + 3 * d.emb * d.mlp + 2 * d.emb


def even_split(layers: int, num_stages: int) -> list[int]:
  base, extra = divmod(layers, num_stages)
  return [base + (1 if i < extra else 0) for i in range(num_stages)]


def stage_flops(d: Dims, seq_len: int, stage_layers, *, remat: bool = True) -> list[dict]:
  """Model and executed FLOPs per token of each stage."""
  layer = sum(layer_forward(d, seq_len).values())
  out = []
  for i, n in enumerate(stage_layers):
    forward = n * layer + (head_forward(d) if i == len(stage_layers) - 1 else 0)
    model = 3 * forward
    out.append({"stage": i, "layers": n, "model_flops_per_token": model,
                "executed_flops_per_token": model + (n * layer if remat else 0)})
  return out


def recompute_flops_per_token(cfg, seq_len: int, remat: bool) -> float:
  """FLOPs per token that rematerialization adds: every decoder layer's forward again."""
  d = Dims.of(cfg)
  return d.layers * sum(layer_forward(d, seq_len).values()) if remat else 0.0


def balanced_split(d: Dims, seq_len: int, num_stages: int, *, remat: bool = True) -> list[int]:
  """The layer split with the smallest largest stage, in executed FLOPs."""
  best, best_max = None, None
  for last in range(1, d.layers - num_stages + 2):
    split = even_split(d.layers - last, num_stages - 1) + [last] if num_stages > 1 else [d.layers]
    if min(split) < 1:
      continue
    worst = max(s["executed_flops_per_token"] for s in stage_flops(d, seq_len, split, remat=remat))
    if best_max is None or worst < best_max:
      best, best_max = split, worst
  return best


def stage_params(d: Dims, stage_layers) -> list[int]:
  out = []
  for i, n in enumerate(stage_layers):
    p = n * layer_params(d)
    if i == 0:
      p += d.vocab * d.emb
    if i == len(stage_layers) - 1:
      p += d.vocab * d.emb + d.emb
    out.append(p)
  return out


def communication(d: Dims, seq_len: int, stage_layers, *, microbatches: int,
                  microbatch_size: int, devices: int, tensor_parallel: int = 1,
                  reduce_grads: str = "step", stage_fsdp: bool = False,
                  remat: bool = True) -> list[dict]:
  """Bytes each GPU of each stage sends per step, by kind."""
  num_stages = len(stage_layers)
  tp = tensor_parallel
  dp = devices // num_stages // tp
  reductions = microbatches if reduce_grads == "microbatch" else 1
  local_tokens = microbatch_size // dp * seq_len
  out = []
  for i, (n, p) in enumerate(zip(stage_layers, stage_params(d, stage_layers))):
    # A ring all-reduce, or with FSDP an all-gather of the weights and a reduce-scatter
    # of the gradients: the same bytes either way.
    grads = 2 * (dp - 1) / dp * (p / tp * PARAM_BYTES) * reductions
    # Two all-reduces per layer forward (attention and MLP outputs), two backward, and
    # two more for the rematerialized forward.
    tp_bytes = (n * microbatches * (6 if remat else 4) * 2 * (tp - 1) / tp
                * local_tokens * d.emb * ACT_BYTES) if tp > 1 else 0.0
    boundary = microbatches * local_tokens * d.emb * ACT_BYTES
    pipeline = boundary * ((i < num_stages - 1) + (i > 0))
    out.append({"stage": i, "gradient_bytes": grads, "tensor_parallel_bytes": tp_bytes,
                "pipeline_bytes": pipeline, "data_parallel": dp, "params": p,
                "fsdp": stage_fsdp})
  return out


def model_dims(model: str):
  """`Dims` and the LlamaConfig of a JAX-Titan flavor or `maxtext:<name>`."""
  if model.startswith("maxtext:"):
    from jaxtitan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top

    cfg = mt.llama_config(mt.config(model.split(":", 1)[1]))
  else:
    from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top

    cfg = llama.FLAVORS[model]
  return Dims.of(cfg), cfg


def report(model: str, *, num_stages: int = 2, stage_layers=None, seq_len: int = 2048,
           microbatches: int = 8, microbatch_size: int = 4, devices: int = 8,
           tensor_parallel: int = 1, reduce_grads: str = "step", stage_fsdp: bool = False,
           remat: bool = True, peak_tflops: float = 2500.0, busbw_gbs: float | None = None) -> dict:
  d, _ = model_dims(model)
  stage_layers = list(stage_layers or even_split(d.layers, num_stages))
  tokens = microbatches * microbatch_size * seq_len
  gpus_per_stage = devices // num_stages
  stages = stage_flops(d, seq_len, stage_layers, remat=remat)
  comm = communication(d, seq_len, stage_layers, microbatches=microbatches,
                       microbatch_size=microbatch_size, devices=devices,
                       tensor_parallel=tensor_parallel, reduce_grads=reduce_grads,
                       stage_fsdp=stage_fsdp, remat=remat)
  for s, c in zip(stages, comm):
    s["model_flops"] = s["model_flops_per_token"] * tokens
    s["executed_flops"] = s["executed_flops_per_token"] * tokens
    s["compute_s_at_peak"] = s["executed_flops"] / (gpus_per_stage * peak_tflops * 1e12)
    s.update(c)
    if busbw_gbs:
      s["gradient_s"] = c["gradient_bytes"] / (busbw_gbs * 1e9)
  ops = {k: v * d.layers for k, v in layer_forward(d, seq_len).items()}
  ops["lm_head"] = head_forward(d)
  model_flops = sum(s["model_flops"] for s in stages)
  return {
      "model": model, "seq_len": seq_len, "tokens_per_step": tokens, "devices": devices,
      "stage_layers": stage_layers, "remat": remat, "peak_tflops": peak_tflops,
      "forward_flops_per_token": ops,
      "model_flops_per_token": model_flops / tokens,
      "model_flops": model_flops,
      "executed_flops": sum(s["executed_flops"] for s in stages),
      "stages": stages,
      "balanced_split": balanced_split(d, seq_len, num_stages, remat=remat),
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--model", default="llama3.1-8b")
  parser.add_argument("--num_stages", type=int, default=2)
  parser.add_argument("--stage_layers", default="")
  parser.add_argument("--seq_len", type=int, default=2048)
  parser.add_argument("--microbatches", type=int, default=8)
  parser.add_argument("--microbatch_size", type=int, default=4)
  parser.add_argument("--devices", type=int, default=8)
  parser.add_argument("--tensor_parallel", type=int, default=1)
  parser.add_argument("--reduce_grads", default="step", choices=["step", "microbatch"])
  parser.add_argument("--stage_fsdp", action="store_true")
  parser.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--peak_tflops", type=float, default=2500.0,
                      help="Per GPU; 1436 is XLA's measured BF16 GEMM rate on MI355X.")
  parser.add_argument("--busbw_gbs", type=float, default=None,
                      help="Bus bandwidth of the gradient collective, e.g. 160.")
  parser.add_argument("--json", default="")
  args = parser.parse_args(argv)
  layers = [int(n) for n in args.stage_layers.split(",")] if args.stage_layers else None
  r = report(args.model, num_stages=args.num_stages, stage_layers=layers, seq_len=args.seq_len,
             microbatches=args.microbatches, microbatch_size=args.microbatch_size,
             devices=args.devices, tensor_parallel=args.tensor_parallel,
             reduce_grads=args.reduce_grads, stage_fsdp=args.stage_fsdp, remat=args.remat,
             peak_tflops=args.peak_tflops, busbw_gbs=args.busbw_gbs)
  print(f"{r['model']}: {r['model_flops_per_token'] / 1e9:.2f} GFLOP per token for training, "
        f"{r['tokens_per_step']} tokens per step; model {r['model_flops'] / 1e15:.2f} PFLOP, "
        f"executed {r['executed_flops'] / 1e15:.2f} PFLOP per step"
        + (" (rematerialized layers)" if r["remat"] else ""))
  print("forward GFLOP per token: " + ", ".join(
      f"{k} {v / 1e9:.2f}" for k, v in r["forward_flops_per_token"].items()))
  print("stage  layers  model TFLOP  executed TFLOP  ms at peak  grad GB/GPU  pipeline GB/GPU  "
        "TP GB/GPU" + ("  grad ms" if args.busbw_gbs else ""))
  for s in r["stages"]:
    line = (f"{s['stage']:5d}  {s['layers']:6d}  {s['model_flops'] / 1e12:11.1f}  "
            f"{s['executed_flops'] / 1e12:14.1f}  {1e3 * s['compute_s_at_peak']:10.1f}  "
            f"{s['gradient_bytes'] / 1e9:11.2f}  {s['pipeline_bytes'] / 1e9:15.3f}  "
            f"{s['tensor_parallel_bytes'] / 1e9:9.2f}")
    if args.busbw_gbs:
      line += f"  {1e3 * s['gradient_s']:7.1f}"
    print(line)
  print(f"balanced split (executed FLOPs): --stage_layers={','.join(map(str, r['balanced_split']))}")
  if args.json:
    with open(args.json, "w") as f:
      json.dump(r, f, indent=1)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
