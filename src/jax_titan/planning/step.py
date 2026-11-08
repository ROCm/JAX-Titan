"""Step time of a training layout, part by part: compute, collectives, pipeline bubble,
runtime gaps.

  python -m jax_titan.planning.step --model=llama3.1-8b --gpus=8 --pp=2 --seq_len=2048
  python -m jax_titan.planning.step --calibrate runs/a [--perfdb perf.jsonl]
  python -m jax_titan.planning.step --calibrate runs/a --ladder 8,64,256,1024 [--weak]

One optimizer step on the busiest GPU (the last pipeline stage) is

  compute + bubble + the exposed part of each collective + runtime gaps

- compute: the stage's executed FLOPs per GPU over an attainable rate, the GEMM rate
  XLA reaches on the GPU (`jax_titan.observability.ceilings`) times `--compute_efficiency`,
  the share of it whole kernels reach;
- collectives, one row per parallelism with its library, operation, group, nodes,
  message size, calls per step, algorithm and bus bandwidth (`planning.collectives`):
  data-parallel gradient all-reduces, FSDP's weight all-gathers and gradient
  reduce-scatters, tensor-parallel all-reduces of activations, pipeline sends and
  receives, and MORI-EP's dispatch and combine for expert parallelism. Each kind hides
  its `--overlap` share (kind=fraction) behind compute, at most as long as the compute;
- bubble: (stages - 1) / microbatches of the compute, a third of it for the zero-bubble
  H1 schedule;
- runtime gaps: idle time beyond the bubble, such as kernel launches and host work; zero
  unless measured.

`--calibrate RUN` predicts the layout a JAX-Titan run used and sets each part against
what its profile measured, then fits the compute efficiency, the runtime gaps and the
collectives' overlap to the run, so that `--ladder` predicts other GPU counts from the
measured parts rather than from one MFU. Fix the part that disagrees, not the total.
"""

import argparse
import dataclasses
import json
import math
import pathlib

from jax_titan.planning import collectives
from jax_titan.planning import gpus as gpus_lib
from jax_titan.planning import memory
from jax_titan.planning import models

# BF16 GEMM rates XLA reaches (jax_titan.observability.ceilings, docs/metrics.md); other
# GPUs default to the same share of their peak.
GEMM_CEILING_TFLOPS = {"AMD Instinct MI355X": 1436.0}
GEMM_CEILING_SHARE = 1436.0 / 2516.6
BUBBLE = {"gpipe": 1.0, "1f1b": 1.0, "zb-h1": 1 / 3, "zb-v": 0.0}
SCHEDULES = {"GPIPE": "gpipe", "ONE_FWD_ONE_BWD": "1f1b", "ZERO_BUBBLE_H1": "zb-h1",
             "CIRCULAR": "1f1b"}
OVERLAP = {"dp": 0.0, "fsdp": 0.9, "tp": 0.0, "pp": 0.9, "ep": 0.0}


@dataclasses.dataclass
class Layout:
  gpus: int
  tp: int = 1
  pp: int = 1
  ep: int = 1
  fsdp: bool = False
  microbatches: int = 8
  microbatch_size: float = 4  # sequences per microbatch, over the data-parallel ranks
  seq_len: int = 2048
  remat: bool = True
  schedule: str = "zb-h1"
  reduce_grads: str = "step"  # or "microbatch"
  gpus_per_node: int = 8

  @property
  def dp(self) -> int:
    return self.gpus // (self.tp * self.pp)

  @property
  def tokens_per_step(self) -> float:
    return self.microbatches * self.microbatch_size * self.seq_len


def _group_nodes(ranks: int, stride: int, gpus_per_node: int) -> tuple[int, int]:
  """Ranks per node and nodes of a group whose members sit `stride` GPUs apart."""
  per_node = max(1, min(ranks, gpus_per_node // stride))
  return per_node, math.ceil(ranks / per_node)


def _layer_params(model: models.Model, tp: int, ep: int) -> float:
  """Parameters per GPU of a decoder layer, averaged over dense and MoE layers."""
  dense = model.attention_params() + model.mlp_params(False) + 2 * model.emb
  moe = (model.attention_params() + 2 * model.emb + model.emb * model.experts
         + model.shared_experts * model.expert_params()
         + model.experts * model.expert_params() / ep)
  return (model.dense_layers * dense + model.moe_layers * moe) / model.layers / tp


def predict(model: models.Model, gpu: gpus_lib.GPU, lo: Layout, *, compute_efficiency=1.0,
            compute_tflops=None, runtime_s=0.0, overlap=None, grad_bytes=4, weight_bytes=2,
            act_bytes=2, dispatch_bytes=1, ep_library="mori", table=None,
            scaleout_gbs=None) -> dict:
  overlap = {**OVERLAP, **(overlap or {})}
  m, dp = lo.microbatches, lo.dp
  stage_layers = model.layers / lo.pp
  forward = model.forward_flops_per_token(lo.seq_len)
  layer = (sum(forward.values()) - forward["lm_head"]) / model.layers
  stage_forward = stage_layers * layer + forward["lm_head"]
  executed = 3 * stage_forward + (stage_layers * layer if lo.remat else 0.0)
  flops_per_gpu = executed * lo.tokens_per_step / (dp * lo.tp)
  rate = (compute_tflops or GEMM_CEILING_TFLOPS.get(gpu.name)
          or gpu.bf16_tflops * GEMM_CEILING_SHARE) * 1e12
  compute_s = flops_per_gpu / (rate * compute_efficiency)

  tokens = lo.microbatch_size * lo.seq_len / dp  # per data-parallel rank and microbatch
  gpn = lo.gpus_per_node
  rows = []

  def add(kind, op, ranks, nodes, nbytes, calls, library="rccl"):
    if ranks < 2 or calls <= 0:
      return
    c = collectives.predict(op, nbytes, ranks, nodes, gpu=gpu, library=library, table=table,
                            scaleout_gbs=scaleout_gbs)
    rows.append({"kind": kind, "library": library, "op": op, "ranks": ranks, "nodes": nodes,
                 "bytes": nbytes, "calls": calls, "algorithm": c["algorithm"],
                 "time_us": c["time_us"], "busbw_gbs": c["busbw_gbs"],
                 "total_s": calls * c["time_us"] / 1e6, "source": c["source"]})

  lay = memory.Layout(data_parallel=dp, tensor_parallel=lo.tp, pipeline_stages=lo.pp,
                      expert_parallel=lo.ep, fsdp=lo.fsdp, microbatches=m,
                      microbatch_size=lo.microbatch_size, seq_len=lo.seq_len, remat=lo.remat)
  stage_params = memory.per_gpu(model, lay)["params_per_gpu"]
  dp_k, dp_nodes = _group_nodes(dp, lo.tp, gpn) if dp * lo.tp > gpn else (dp, 1)
  if lo.fsdp:
    per_layer = _layer_params(model, lo.tp, lo.ep)
    gathers = stage_layers * (m if lo.pp > 1 else 1)
    add("fsdp", "all_gather", dp, dp_nodes, per_layer * weight_bytes, 2 * gathers)
    add("fsdp", "reduce_scatter", dp, dp_nodes, per_layer * grad_bytes, gathers)
  else:
    add("dp", "all_reduce", dp, dp_nodes, stage_params * grad_bytes,
        m if lo.reduce_grads == "microbatch" else 1)
  activation = tokens * model.emb * act_bytes
  tp_nodes = math.ceil(lo.tp / gpn)
  add("tp", "all_reduce", lo.tp, tp_nodes, activation,
      stage_layers * m * (6 if lo.remat else 4))
  add("pp", "send_recv", 2 if lo.pp > 1 else 1, 1 if lo.gpus <= gpn else 2, activation, 2 * m)
  if model.moe_layers and lo.ep > 1:
    ep_k, ep_nodes = _group_nodes(lo.ep, lo.tp, gpn) if lo.ep * lo.tp > gpn else (lo.ep, 1)
    moe_calls = model.moe_layers / lo.pp * m * (3 if lo.remat else 2)
    routed = tokens * model.experts_per_token * model.emb
    add("ep", "dispatch", lo.ep, ep_nodes, routed * dispatch_bytes, moe_calls, ep_library)
    add("ep", "combine", lo.ep, ep_nodes, routed * act_bytes, moe_calls, ep_library)

  exposed = {}
  for r in rows:
    exposed[r["kind"]] = exposed.get(r["kind"], 0.0) + r["total_s"]
  # A kind hides its `overlap` share behind compute, and no more than the compute lasts.
  exposed = {k: max(v * (1 - overlap.get(k, 0.0)), v - compute_s) for k, v in exposed.items()}
  bubble_s = BUBBLE[lo.schedule] * (lo.pp - 1) / m * compute_s if lo.pp > 1 else 0.0
  step_s = compute_s + bubble_s + sum(exposed.values()) + runtime_s
  model_flops = model.training_flops_per_token(lo.seq_len) * lo.tokens_per_step
  return {"layout": dataclasses.asdict(lo), "dp": dp, "nodes": math.ceil(lo.gpus / gpn),
          "compute_s": compute_s, "bubble_s": bubble_s, "runtime_s": runtime_s,
          "exposed_s": exposed, "collectives": rows, "step_s": step_s,
          "tokens_per_s": lo.tokens_per_step / step_s,
          "mfu": model_flops / (step_s * lo.gpus * gpu.bf16_tflops * 1e12),
          "compute_efficiency": compute_efficiency, "gemm_tflops": rate / 1e12}


def run_layout(start: dict) -> Layout:
  """The layout of a `jax_titan.trainer.train` run, from its start event."""
  a = start["args"]
  mpmd = a.get("parallel", "mpmd") == "mpmd"
  pp = a.get("num_stages", 1) if mpmd else 1
  microbatches, size = a.get("microbatches", 8), a.get("microbatch_size", 4)
  if not mpmd:
    microbatches, size = 1, microbatches * size
  return Layout(
      gpus=start["num_devices"], tp=a.get("tensor_parallel", 1), pp=pp,
      fsdp=bool(a.get("stage_fsdp")) or a.get("parallel") == "fsdp",
      microbatches=microbatches, microbatch_size=size, seq_len=a.get("seq_len", 2048),
      remat=a.get("remat", True),
      schedule=SCHEDULES.get(str(a.get("schedule") or "ONE_FWD_ONE_BWD"), "1f1b"),
      reduce_grads=a.get("reduce_grads") or ("microbatch" if ":" in a["model"] else "step"))


def measured_parts(run_dir) -> dict:
  """Step time and, on the busiest GPU of the run's profile, its parts per step."""
  from jax_titan.observability import records  # pylint: disable=g-import-not-at-top
  from jax_titan.observability import summarize  # pylint: disable=g-import-not-at-top

  run = pathlib.Path(run_dir)
  metrics = summarize.run_metrics(run)
  out = {"step_s": metrics["step_time_s"]["p50"], "mfu": metrics.get("mfu")}
  profile = run / "profile.json"
  if profile.exists():
    p = json.loads(profile.read_text())
    per_step = p["window_s"] / (p.get("steps") or 1)
    busiest = max(p["gpus"].values(), key=lambda g: g["compute"])
    out.update({k: busiest[k] * per_step for k in ("compute", "collective", "copy", "idle")})
  log = records.read(run / "train.jsonl")
  out["start"] = records.events(log, "start")[0]
  return out


def calibrate(model, gpu, lo: Layout, measured: dict, **kwargs) -> dict:
  """Predicted against measured, part by part, and the fitted efficiency, gaps and
  overlap that reproduce the run."""
  base = predict(model, gpu, lo, **kwargs)
  out = {"predicted": base, "measured": measured}
  if "compute" not in measured:
    return out
  efficiency = base["compute_s"] * base["compute_efficiency"] / measured["compute"]
  comm = sum(r["total_s"] for r in base["collectives"])
  hidden = 1 - measured["collective"] / comm if comm > measured["collective"] else 0.0
  # The run measures the kinds that carried its traffic; the others keep their defaults.
  totals = {}
  for r in base["collectives"]:
    totals[r["kind"]] = totals.get(r["kind"], 0.0) + r["total_s"]
  overlap = {**OVERLAP, **(kwargs.get("overlap") or {}),
             **{k: hidden for k, v in totals.items() if v > 0.1 * comm}}
  fit_kwargs = {**kwargs, "compute_efficiency": efficiency, "overlap": overlap}
  runtime = max(0.0, measured["idle"] - predict(model, gpu, lo, **fit_kwargs)["bubble_s"])
  fit_kwargs["runtime_s"] = runtime + measured.get("copy", 0.0)
  fitted = predict(model, gpu, lo, **fit_kwargs)
  parts = [("compute", base["compute_s"], measured["compute"]),
           ("exposed collectives", sum(base["exposed_s"].values()), measured["collective"]),
           ("bubble and gaps", base["bubble_s"] + base["runtime_s"],
            measured["idle"] + measured.get("copy", 0.0)),
           ("step", base["step_s"], measured["step_s"])]
  out.update(parts=[{"part": n, "predicted_s": p, "measured_s": m_, "ratio": m_ / p if p else None}
                    for n, p, m_ in parts],
             fit={"compute_efficiency": efficiency, "overlap": overlap, "hidden": hidden,
                  "runtime_s": fit_kwargs["runtime_s"]},
             fitted=fitted)
  worst = max(out["parts"][:3], key=lambda r: abs(r["measured_s"] - r["predicted_s"]))
  out["bottleneck"] = worst["part"]
  return out


def ladder(model, gpu, lo: Layout, counts, *, weak=False, batch_tokens=None,
           **kwargs) -> list[dict]:
  """The layout's shape at each GPU count, with the same global batch (strong scaling,
  `batch_tokens` or the layout's) or the same batch per GPU (weak). Runtime gaps are a
  cost per microbatch; counts that leave a data-parallel rank less than one sequence
  per microbatch are skipped."""
  out = []
  base = dataclasses.replace(lo, microbatch_size=batch_tokens / (lo.microbatches * lo.seq_len)
                             ) if batch_tokens else lo
  for n in counts:
    if n % (lo.tp * lo.pp):
      continue
    size = base.microbatch_size * n / base.gpus if weak else base.microbatch_size
    step = dataclasses.replace(base, gpus=n, microbatch_size=size)
    if step.microbatch_size / step.dp < 1:
      continue
    r = predict(model, gpu, step, **kwargs)
    r["gpus"] = n
    out.append(r)
  return out


def _print_step(r: dict) -> None:
  lo = r["layout"]
  print(f"{lo['gpus']} GPUs on {r['nodes']} node(s): dp {r['dp']}, tp {lo['tp']}, pp "
        f"{lo['pp']}, ep {lo['ep']}{', fsdp' if lo['fsdp'] else ''}; {lo['microbatches']} "
        f"microbatches of {lo['microbatch_size']:g} sequences of {lo['seq_len']}, "
        f"{lo['schedule']}")
  print(f"{'kind':5s} {'library':7s} {'op':15s} {'ranks':>5s} {'nodes':>5s} {'message':>10s} "
        f"{'calls':>7s} {'algorithm':>13s} {'us each':>9s} {'busbw':>7s} {'s/step':>7s}")
  for c in r["collectives"]:
    print(f"{c['kind']:5s} {c['library']:7s} {c['op']:15s} {c['ranks']:5d} {c['nodes']:5d} "
          f"{collectives._bytes_label(c['bytes']):>10s} {c['calls']:7.0f} {c['algorithm']:>13s} "
          f"{c['time_us']:9.1f} {c['busbw_gbs']:7.1f} {c['total_s']:7.3f}")
  exposed = ", ".join(f"{k} {v:.3f}" for k, v in r["exposed_s"].items()) or "none"
  print(f"compute {r['compute_s']:.3f} s (at {r['compute_efficiency']:.0%} of "
        f"{r['gemm_tflops']:.0f} TFLOP/s), bubble {r['bubble_s']:.3f} s, exposed collectives "
        f"{exposed}, runtime gaps {r['runtime_s']:.3f} s")
  print(f"step {r['step_s']:.3f} s, {r['tokens_per_s'] / 1e3:.1f}k tokens/s, MFU "
        f"{100 * r['mfu']:.1f}%")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--model", default="")
  parser.add_argument("--gpu", default="MI355X")
  parser.add_argument("--gpus", type=int, default=8)
  parser.add_argument("--gpus_per_node", type=int, default=0,
                      help="Default: the GPU's scale-up domain (8, or 72 for MI455X).")
  parser.add_argument("--tp", type=int, default=1)
  parser.add_argument("--pp", type=int, default=1)
  parser.add_argument("--ep", type=int, default=1)
  parser.add_argument("--fsdp", action="store_true")
  parser.add_argument("--microbatches", type=int, default=8)
  parser.add_argument("--microbatch_size", type=float, default=4)
  parser.add_argument("--seq_len", type=int, default=2048)
  parser.add_argument("--no-remat", dest="remat", action="store_false")
  parser.add_argument("--schedule", default="zb-h1", choices=list(BUBBLE))
  parser.add_argument("--reduce_grads", default="step", choices=["step", "microbatch"])
  parser.add_argument("--compute_efficiency", type=float, default=1.0)
  parser.add_argument("--compute_tflops", type=float, default=None)
  parser.add_argument("--runtime_s", type=float, default=0.0)
  parser.add_argument("--overlap", default="",
                      help="kind=fraction,...: dp, fsdp, tp, pp, ep (default "
                      + ",".join(f"{k}={v:g}" for k, v in OVERLAP.items()) + ").")
  parser.add_argument("--ep_library", default="mori", choices=["mori", "rccl"])
  parser.add_argument("--dispatch_bytes", type=int, default=1, help="1: FP8 dispatch.")
  parser.add_argument("--comm_table", default="", help="Measured collectives (--calibrate).")
  parser.add_argument("--scaleout_gbs", type=float, default=None)
  parser.add_argument("--calibrate", default="", help="A JAX-Titan --run_dir.")
  parser.add_argument("--ladder", default="", help="GPU counts, e.g. 8,64,256,1024.")
  parser.add_argument("--weak", action="store_true",
                      help="--ladder: grow the batch with the GPUs instead of keeping it.")
  parser.add_argument("--ladder_batch_tokens", type=float, default=None,
                      help="--ladder: the global batch to scale at, e.g. 4194304 (default: "
                      "the layout's).")
  parser.add_argument("--perfdb", default="",
                      help="--calibrate: append predicted and measured to this JSONL file.")
  parser.add_argument("--json", default="")
  args = parser.parse_args(argv)

  gpu = gpus_lib.get(args.gpu)
  gpn = args.gpus_per_node or gpu.scaleup_gpus or 8
  measured = None
  if args.calibrate:
    measured = measured_parts(args.calibrate)
    lo = dataclasses.replace(run_layout(measured["start"]), gpus_per_node=gpn)
    model = models.from_name(args.model or measured["start"]["args"]["model"])
  else:
    if not args.model:
      parser.error("--model or --calibrate is required")
    model = models.from_name(args.model)
    lo = Layout(args.gpus, args.tp, args.pp, args.ep, args.fsdp, args.microbatches,
                args.microbatch_size, args.seq_len, args.remat, args.schedule,
                args.reduce_grads, gpn)
  overlap = {k: float(v) for k, v in (kv.split("=") for kv in args.overlap.split(",") if kv)}
  kwargs = dict(compute_efficiency=args.compute_efficiency, compute_tflops=args.compute_tflops,
                runtime_s=args.runtime_s, overlap=overlap, dispatch_bytes=args.dispatch_bytes,
                ep_library=args.ep_library, scaleout_gbs=args.scaleout_gbs,
                table=collectives.Table.load(args.comm_table) if args.comm_table else None)
  out = {}
  if measured:
    c = calibrate(model, gpu, lo, measured, **kwargs)
    out["calibration"] = {k: v for k, v in c.items() if k != "measured"}
    _print_step(c["predicted"])
    if "parts" in c:
      print(f"\n{'part':20s} {'predicted s':>11s} {'measured s':>10s} {'measured/predicted':>18s}")
      for p in c["parts"]:
        print(f"{p['part']:20s} {p['predicted_s']:11.3f} {p['measured_s']:10.3f} "
              f"{p['ratio']:18.2f}")
      f = c["fit"]
      print(f"largest gap: {c['bottleneck']}. Fitted to the run: kernels at "
            f"{f['compute_efficiency']:.0%} of the GEMM rate, {f['hidden']:.0%} of the "
            f"measured collectives hidden, {f['runtime_s']:.3f} s of runtime gaps and copies "
            f"per step; fitted step {c['fitted']['step_s']:.3f} s")
      kwargs.update(compute_efficiency=f["compute_efficiency"], overlap=f["overlap"],
                    runtime_s=f["runtime_s"])
      if args.perfdb:
        from jax_titan.observability import perfdb  # pylint: disable=g-import-not-at-top

        perfdb.append(args.perfdb, perfdb.step_record(args.calibrate, model.name, gpu.name, lo,
                                                      c))
  else:
    out["step"] = predict(model, gpu, lo, **kwargs)
    _print_step(out["step"])
  if args.ladder:
    counts = [int(n) for n in args.ladder.split(",")]
    rows = ladder(model, gpu, lo, counts, weak=args.weak, batch_tokens=args.ladder_batch_tokens,
                  **kwargs)
    out["ladder"] = rows
    print(f"\n{'scaling' if not args.weak else 'weak scaling'} of this layout's shape"
          + (" with the fit" if measured and "parts" in c else "") + ":")
    print(f"{'GPUs':>6s} {'nodes':>5s} {'dp':>5s} {'compute':>8s} {'bubble':>7s} "
          f"{'exposed comm':>24s} {'step s':>7s} {'MFU':>6s} {'tokens/s':>10s}")
    for r in rows:
      comm = " ".join(f"{k} {v:.3f}" for k, v in r["exposed_s"].items() if v > 0.0005) or "-"
      print(f"{r['gpus']:6d} {r['nodes']:5d} {r['dp']:5d} {r['compute_s']:8.3f} "
            f"{r['bubble_s']:7.3f} {comm:>24s} {r['step_s']:7.3f} {100 * r['mfu']:5.1f}% "
            f"{r['tokens_per_s']:10.0f}")
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
