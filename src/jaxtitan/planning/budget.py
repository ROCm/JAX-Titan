"""Training budget: compute, time, GPU-hours, energy, memory and communication of a run.

  python -m jaxtitan.planning.budget --model=llama3.1-8b --tokens=1e12 --gpu=MI355X --gpus=1024
  python -m jaxtitan.planning.budget --model=maxtext:llama3.1-70b --tokens=15e12 \
      --gpu=MI300X,MI325X,MI355X,MI455X --gpus=2048
  python -m jaxtitan.planning.budget --model=hf:config.json --days=30 --gpu=MI325X --gpus=1024
  python -m jaxtitan.planning.budget --model=maxtext:llama3.1-8b --tokens=2e12 --gpu=MI355X \
      --gpus=512 --from_run=runs/maxtext-llama3.1-8b

`--model` is a JAX-Titan Llama flavor, `maxtext:<name>` or `hf:<config.json>`, dense or
mixture-of-experts. Give the tokens to train (`--tokens`, or `--dataset_tokens` and
`--epochs`) or a time budget (`--days`, `--gpu_hours`). One `--gpu` gets a full report;
a comma-separated list gets a table with a row per GPU. It reports:

- training FLOPs: model FLOPs (three forward passes per token of the active parameters,
  plus causal attention; the numerator of MFU) and executed FLOPs, which add the forward
  pass that rematerialization repeats; `--attention=none` gives the 6N convention of
  published compute figures, `--attention=full` PaLM's and Megatron's;
- throughput and time at `--mfu` of the GPU's dense peak at `--precision`, or at what a
  JAX-Titan run measured (`--from_run`: its achieved FLOP/s on the same GPU type, its MFU
  carried over to other types), stretched by ETTR for checkpoints, restarts and lost work:
  `--ettr`, or projected for this many GPUs from a checkpoint cost, a restart cost and a
  failure rate (defaults: the MaxText Llama 3.1 8B recipe, and Llama 3's 419 unexpected
  interruptions in 54 days on 16,384 GPUs);
- GPU-hours, energy at the GPUs' board power times `--power_fraction` and `--pue` (the
  GPU boundary; a run's power.jsonl gives the measured fraction), and cost;
- memory per GPU of the layout (`--tp`, `--pp`, `--ep`, `--fsdp`, `--batch_tokens`,
  `--microbatches`), whether it fits the GPU's HBM, and the layouts that do;
- the data-parallel gradient reduction (or FSDP's gathers) per step against the step's
  compute, and the batch that hides it;
- the compute-optimal model size and tokens for the same compute (Chinchilla, about 20
  tokens per parameter with C = 6ND).

MFU measured on one GPU type is only an assumption on another; time scales with 1 / MFU.
"""

import argparse
import dataclasses
import json
import math

from jaxtitan.planning import comms
from jaxtitan.planning import gpus as gpus_lib
from jaxtitan.planning import memory
from jaxtitan.planning import models

DAY = 86400.0
CHINCHILLA_TOKENS_PER_PARAM = 20
# Measured with JAX-Titan on Llama 3.1 8B (docs/planning.md): the recipe's steady checkpoint
# cost and restart cost including the first save, and peak HBM over the analytic floor of
# JAX-Titan's own pipeline (1.96-2.01; the MaxText adapter's 2.83).
RECIPE_W_CP_S = 2.4
RECIPE_RESTART_S = 175.0
MEASURED_MEMORY_OVERHEAD = 2.0
# Llama 3 (arXiv:2407.21783): 419 unexpected interruptions in 54 days on 16,384 GPUs.
LLAMA3_FAILURES_PER_1000_GPU_DAYS = 1000 * 419 / (54 * 16384)
# BF16 GEMM rates XLA reaches (jaxtitan.observability.ceilings; docs/metrics.md): an upper
# bound on MFU end to end.
MEASURED_BF16_GEMM_TFLOPS = {"AMD Instinct MI355X": 1436.0}
SENSITIVITY_MFUS = (0.1, 0.2, 0.3, 0.4, 0.5)


def _duration(days: float) -> str:
  if days >= 365.25:
    return f"{days / 365.25:,.1f} years"
  if days >= 1:
    return f"{days:.1f} days"
  return f"{days * 24:.1f} hours"


def _count(x: float) -> str:
  for scale, unit in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
    if x >= scale:
      return f"{x / scale:.2f}{unit}"
  return f"{x:.0f}"


def projected_ettr(gpus: int, w_cp_s: float, restart_s: float,
                   failures_per_1000_gpu_days: float) -> tuple[float, float]:
  """ETTR at Daly's checkpoint interval, and that interval in seconds."""
  from jaxtitan.observability import ettr  # pylint: disable=g-import-not-at-top

  mttf = 1000.0 / (failures_per_1000_gpu_days * gpus) * DAY
  interval = ettr.daly_interval(w_cp_s, mttf)
  return ettr.expected_ettr(w_cp_s, restart_s, mttf, interval), interval


def estimate(model: models.Model, gpu: gpus_lib.GPU, gpus: int, *, tokens: float | None = None,
             seconds: float | None = None, mfu: float = 0.4, ettr: float = 1.0,
             precision: str = "bf16", seq_len: int = 8192, remat: bool = True,
             attention: str = "causal", power_fraction: float = 1.0, pue: float = 1.0,
             usd_per_gpu_hour: float | None = None) -> dict:
  """Time and cost of training `tokens`, or the tokens `seconds` of wall time buy."""
  if (tokens is None) == (seconds is None):
    raise ValueError("give tokens or seconds")
  peak = gpu.peak_tflops(precision) * 1e12
  per_token = model.training_flops_per_token(seq_len, attention)
  executed_per_token = per_token + model.recompute_flops_per_token(seq_len, remat, attention)
  tokens_per_s = gpus * peak * mfu / per_token
  if tokens is None:
    tokens = tokens_per_s * seconds * ettr
  wall_s = tokens / tokens_per_s / ettr
  gpu_hours = gpus * wall_s / 3600
  model_flops = tokens * per_token
  # Chinchilla's C = 6ND, with N the parameters a token uses.
  n_opt = math.sqrt(6 * model.params(active=True) * tokens / (6 * CHINCHILLA_TOKENS_PER_PARAM))
  out = {
      "model": model.name, "params": model.params(), "active_params": model.params(active=True),
      "gpu": gpu.name, "gpus": gpus, "precision": precision, "peak_tflops": peak / 1e12,
      "seq_len": seq_len, "tokens": tokens, "mfu": mfu,
      "hfu": mfu * executed_per_token / per_token, "ettr": ettr,
      "model_flops_per_token": per_token, "executed_flops_per_token": executed_per_token,
      "model_flops": model_flops, "executed_flops": tokens * executed_per_token,
      "tokens_per_s": tokens_per_s, "tokens_per_s_per_gpu": tokens_per_s / gpus,
      "compute_days": wall_s * ettr / DAY, "wall_days": wall_s / DAY, "gpu_hours": gpu_hours,
      "compute_optimal": {"params": n_opt, "tokens": CHINCHILLA_TOKENS_PER_PARAM * n_opt},
      "tokens_per_param": tokens / model.params(active=True),
  }
  if gpu.board_power_w:
    out["energy_mwh"] = gpus * gpu.board_power_w * power_fraction * wall_s / 3600 / 1e6 * pue
    out["power_fraction"], out["pue"] = power_fraction, pue
  if usd_per_gpu_hour:
    out["cost_usd"] = gpu_hours * usd_per_gpu_hour
  return out


def layouts_that_fit(model: models.Model, gpu: gpus_lib.GPU, gpus: int, base: memory.Layout,
                     overhead: float = MEASURED_MEMORY_OVERHEAD, margin: float = 0.95,
                     limit: int = 6) -> list[dict]:
  """Layouts whose memory (floor x overhead) fits margin x HBM, fewest model-parallel
  ways first."""
  if not gpu.hbm_gb:
    return []
  out = []
  eps = [1] + ([e for e in (2, 4, 8, 16, 32, 64) if model.experts and model.experts % e == 0]
               if model.moe_layers else [])
  for tp in (1, 2, 4, 8):
    for pp in (1, 2, 4, 8, 16, 32):
      if gpus % (tp * pp) or pp > model.layers:
        continue
      for fsdp in (False, True):
        for ep in eps:
          dp = gpus // (tp * pp)
          if ep > dp:
            continue
          lo = dataclasses.replace(base, data_parallel=dp, tensor_parallel=tp, pipeline_stages=pp,
                                   expert_parallel=ep, fsdp=fsdp,
                                   microbatches=max(base.microbatches, pp))
          need = memory.per_gpu(model, lo)["total_bytes"] * overhead
          if need <= margin * gpu.hbm_gb * 1e9:
            out.append({"tp": tp, "pp": pp, "ep": ep, "fsdp": fsdp, "dp": dp,
                        "gib_per_gpu": need / 2**30})
  out.sort(key=lambda r: (r["tp"] * r["pp"], r["fsdp"], r["ep"]))
  return out[:limit]


def from_run(run_dir: str) -> dict:
  """What a JAX-Titan --run_dir measured: MFU and achieved FLOP/s per GPU, mean power per
  GPU, checkpoint and restart costs, and memory over the analytic floor."""
  from jaxtitan.observability import ettr as ettr_lib  # pylint: disable=g-import-not-at-top
  from jaxtitan.observability import records  # pylint: disable=g-import-not-at-top
  from jaxtitan.observability import summarize  # pylint: disable=g-import-not-at-top

  run = summarize.run_metrics(run_dir)
  out = {"run": run_dir, "device_kind": run.get("device_kind"),
         "mfu": run.get("mfu_sustained") or run["mfu"]}
  devices = run.get("devices") or 1
  if run.get("peak_flops_per_s"):
    peak_per_gpu = run["peak_flops_per_s"] / devices
  else:
    log = records.read(f"{run_dir}/train.jsonl")
    start = records.events(log, "start")[0]
    peak_per_gpu = start["args"].get("peak_tflops_per_device", 2500.0) * 1e12
  out["achieved_tflops_per_gpu"] = out["mfu"] * peak_per_gpu / 1e12
  if run.get("energy"):
    out["power_w_per_gpu"] = run["energy"]["mean_power_w"] / devices
  try:
    measured = ettr_lib.measure(run_dir)
    if measured.get("w_cp_s"):
      out["w_cp_s"] = measured["w_cp_s"]
    restarts = [s.get("total", 0.0) for s in measured["u0_stages_s"]]
    if restarts:
      out["restart_s"] = max(restarts)
  except (ValueError, KeyError):
    pass
  try:
    out["memory_overhead"] = memory.calibrate(run_dir)["overhead"]
  except (ValueError, KeyError, NotImplementedError):
    pass
  return out


def plan(model: models.Model, gpu: gpus_lib.GPU, args, run: dict | None = None) -> dict:
  """Everything the report shows for one GPU type."""
  mfu, mfu_source = args.mfu, "assumed"
  power_fraction = args.power_fraction
  if run:
    if run["device_kind"] == gpu.name:
      mfu = run["achieved_tflops_per_gpu"] / gpu.peak_tflops(args.precision)
      mfu_source = f"measured on {gpu.name}"
      if run.get("power_w_per_gpu") and gpu.board_power_w:
        power_fraction = run["power_w_per_gpu"] / gpu.board_power_w
    else:
      mfu, mfu_source = run["mfu"], f"carried over from {run['device_kind']}"
  w_cp = (run or {}).get("w_cp_s", args.w_cp_s)
  restart = (run or {}).get("restart_s", args.restart_s)
  if args.memory_overhead is not None:
    overhead, overhead_source = args.memory_overhead, "given"
  elif run and run.get("memory_overhead"):
    overhead, overhead_source = run["memory_overhead"], "measured"
  else:
    overhead, overhead_source = MEASURED_MEMORY_OVERHEAD, "JAX-Titan Llama 3.1 8B pipeline"

  interval = None
  if args.ettr is None:
    ettr, interval = projected_ettr(args.gpus, w_cp, restart, args.failures_per_1000_gpu_days)
  else:
    ettr = args.ettr
  seconds = args.days * DAY if args.days else (
      args.gpu_hours * 3600 / args.gpus if args.gpu_hours else None)
  tokens = args.dataset_tokens * args.epochs if args.dataset_tokens else args.tokens
  r = estimate(model, gpu, args.gpus, tokens=tokens, seconds=seconds, mfu=mfu, ettr=ettr,
               precision=args.precision, seq_len=args.seq_len, remat=args.remat,
               attention=args.attention, power_fraction=power_fraction, pue=args.pue,
               usd_per_gpu_hour=args.usd_per_gpu_hour)
  r["mfu_source"] = mfu_source
  if interval:
    r["checkpoint_interval_h"] = interval / 3600
  r["wall_days_at_mfu"] = {m: r["wall_days"] * r["mfu"] / m for m in SENSITIVITY_MFUS}
  ceiling = MEASURED_BF16_GEMM_TFLOPS.get(gpu.name) if args.precision == "bf16" else None
  if ceiling:
    r["measured_gemm_tflops"] = ceiling

  dp = max(1, args.gpus // (args.tp * args.pp))
  sequences = args.batch_tokens / args.seq_len
  microbatches = args.microbatches or max(args.pp, math.ceil(sequences / dp))
  weight_bytes, grad_bytes, optimizer_bytes = (int(b) for b in args.state_bytes.split(","))
  layout = memory.Layout(
      data_parallel=dp, tensor_parallel=args.tp, pipeline_stages=args.pp,
      expert_parallel=args.ep, fsdp=args.fsdp,
      microbatch_size=sequences / microbatches, microbatches=microbatches,
      seq_len=args.seq_len, remat=args.remat, weight_bytes=weight_bytes,
      grad_bytes=grad_bytes, optimizer_bytes=optimizer_bytes)
  mem = memory.per_gpu(model, layout)
  hbm = (gpu.hbm_gb or 0) * 1e9
  r["memory"] = {**mem, "layout": dataclasses.asdict(layout), "overhead": overhead,
                 "overhead_source": overhead_source,
                 "with_overhead_bytes": mem["total_bytes"] * overhead, "hbm_bytes": hbm,
                 "fits": mem["total_bytes"] * overhead <= 0.95 * hbm if hbm else None}
  r["layouts_that_fit"] = layouts_that_fit(model, gpu, args.gpus, layout, overhead)
  if not r["layouts_that_fit"] and hbm:
    state = model.params() * (layout.weight_bytes + layout.grad_bytes + layout.optimizer_bytes)
    r["min_gpus_for_state"] = math.ceil(state * overhead / (0.95 * hbm))

  if dp > 1 and gpu.scaleup_gbs:
    k = comms.ranks_per_domain(args.gpus, dp, args.tp, gpu.scaleup_gpus)
    scaleout = args.scaleout_gbs or gpu.scaleout_gbs
    if k < dp and not scaleout:
      r["data_parallel"] = None
    else:
      c = comms.data_parallel_seconds(mem["params_per_gpu"], dp, k, gpu.scaleup_gbs_for(k),
                                      scaleout, grad_bytes=args.grad_reduce_bytes,
                                      fsdp=args.fsdp, efficiency=args.comm_efficiency)
      step_s = args.batch_tokens / r["tokens_per_s"]
      c.update(step_compute_s=step_s, scaleout_gbs=scaleout,
               hiding_batch_tokens=args.batch_tokens * c["total_s"] / step_s)
      r["data_parallel"] = c
  return r


def report(r: dict, args) -> None:
  model_line = (f"{r['model']}: {r['params'] / 1e9:.1f}B parameters"
                + (f" ({r['active_params'] / 1e9:.1f}B active)"
                   if r["active_params"] != r["params"] else "")
                + f", {r['model_flops_per_token'] / 1e9:.1f} GFLOP per token at sequence "
                f"{r['seq_len']}")
  print(model_line)
  print(f"{r['gpus']} x {r['gpu']} at {r['peak_tflops']:.0f} TFLOP/s {r['precision']} dense, "
        f"MFU {100 * r['mfu']:.1f}% ({r['mfu_source']}; HFU {100 * r['hfu']:.1f}%), "
        f"ETTR {r['ettr']:.3f}"
        + (f" (Daly interval {r['checkpoint_interval_h']:.2f} h)"
           if "checkpoint_interval_h" in r else ""))
  print(f"tokens {r['tokens'] / 1e12:.3f}T ({r['tokens_per_param']:.0f} per active parameter): "
        f"model {r['model_flops']:.3e} FLOPs, executed {r['executed_flops']:.3e}")
  print(f"throughput {_count(r['tokens_per_s'])} tokens/s ({r['tokens_per_s_per_gpu']:.0f} "
        f"per GPU); {_duration(r['wall_days'])} wall ({_duration(r['compute_days'])} stepping), "
        f"{_count(r['gpu_hours'])} GPU-hours"
        + (f", {r['energy_mwh']:.0f} MWh at {100 * r['power_fraction']:.0f}% of board power"
           if "energy_mwh" in r else "")
        + (f", ${r['cost_usd'] / 1e6:.2f}M" if "cost_usd" in r else ""))
  print("wall time at MFU " + ", ".join(f"{100 * m:.0f}%: {_duration(d)}"
                                        for m, d in r["wall_days_at_mfu"].items()))
  if "measured_gemm_tflops" in r:
    g = r["measured_gemm_tflops"]
    print(f"XLA reaches {g:.0f} TFLOP/s of BF16 GEMM on {r['gpu']} "
          f"({100 * g / r['peak_tflops']:.0f}% of peak): MFU {100 * r['mfu']:.1f}% is "
          f"{100 * r['mfu'] * r['peak_tflops'] / g:.0f}% of that ceiling")
  co = r["compute_optimal"]
  print(f"compute-optimal for this compute (Chinchilla, C = 6ND): {co['params'] / 1e9:.1f}B "
        f"parameters on {co['tokens'] / 1e12:.2f}T tokens")
  m, lo = r["memory"], r["memory"]["layout"]
  print(f"memory per GPU (tp {lo['tensor_parallel']}, pp {lo['pipeline_stages']}, dp "
        f"{lo['data_parallel']}, ep {lo['expert_parallel']}{', fsdp' if lo['fsdp'] else ''}): "
        f"floor {m['total_bytes'] / 2**30:.0f} GiB (state {m['state_bytes'] / 2**30:.0f}, "
        f"activations {m['activation_bytes'] / 2**30:.0f}, logits "
        f"{m['logits_bytes'] / 2**30:.0f}), x{m['overhead']:.2f} overhead "
        f"({m['overhead_source']}) {m['with_overhead_bytes'] / 2**30:.0f} GiB of "
        f"{m['hbm_bytes'] / 1e9:.0f} GB: "
        + ("fits" if m["fits"] else "does not fit" if m["fits"] is not None else "unknown HBM"))
  for f in r["layouts_that_fit"]:
    print(f"  fits: tp {f['tp']}, pp {f['pp']}, dp {f['dp']}, ep {f['ep']}"
          f"{', fsdp' if f['fsdp'] else ''}: {f['gib_per_gpu']:.0f} GiB")
  if "min_gpus_for_state" in r:
    print(f"  no layout of {r['gpus']} GPUs fits; the training state alone needs at least "
          f"{r['min_gpus_for_state']} GPUs (sharded over all of them, x{m['overhead']:.2f})")
  c = r.get("data_parallel")
  if c:
    what = "FSDP gathers and reduce-scatter" if lo["fsdp"] else "gradient all-reduce"
    print(f"data parallel over {c['domains']} scale-up domain(s) of {c['ranks_per_domain']} "
          f"ranks: {what} of {c['bytes_per_gpu'] / 1e9:.1f} GB per GPU per step, "
          f"{c['inside_s']:.3f} s inside + {c['across_s']:.3f} s across domains at "
          f"{100 * c['efficiency']:.0f}% of peak links vs {c['step_compute_s']:.3f} s of "
          f"compute per {_count(args.batch_tokens)}-token "
          f"step: " + ("hidden if overlapped" if c["total_s"] <= c["step_compute_s"] else
                       f"exposed; hiding it takes {_count(c['hiding_batch_tokens'])} tokens "
                       "per step"))


def comparison(results: list[dict]) -> None:
  """One row per GPU type."""
  print(f"{results[0]['model']}, {results[0]['tokens'] / 1e12:.2f}T tokens on "
        f"{results[0]['gpus']} GPUs ({results[0]['precision']})")
  head = (f"{'GPU':22s} {'peak TF':>8s} {'MFU':>6s} {'tok/s/GPU':>10s} {'wall':>12s} "
          f"{'GPU-hours':>10s} {'MWh':>7s} {'HBM GiB':>15s} {'DP comm/step':>13s}  MFU source")
  print(head)
  print("-" * len(head))
  for r in results:
    m, c = r["memory"], r.get("data_parallel")
    fits = "" if m["fits"] is None else (" ok" if m["fits"] else " no")
    hbm = f"{m['with_overhead_bytes'] / 2**30:.0f}/{m['hbm_bytes'] / 2**30:.0f}{fits}"
    dp = (f"{c['total_s'] / c['step_compute_s']:.2f}x" if c else "-")
    energy = f"{r['energy_mwh']:.0f}" if "energy_mwh" in r else "-"
    print(f"{r['gpu']:22s} {r['peak_tflops']:8.0f} {100 * r['mfu']:5.1f}% "
          f"{r['tokens_per_s_per_gpu']:10.0f} {_duration(r['wall_days']):>12s} "
          f"{_count(r['gpu_hours']):>10s} {energy:>7s} {hbm:>15s} {dp:>13s}  {r['mfu_source']}")
  print("HBM: the layout's floor x overhead over the GPU's HBM; DP comm/step: the data-parallel "
        "collectives over the step's compute, at --comm_efficiency of the links' peak (above "
        "1x stays exposed even when overlapped).")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--model", required=True)
  budget = parser.add_mutually_exclusive_group(required=True)
  budget.add_argument("--tokens", type=float,
                      help="Tokens processed over the whole run (every epoch), e.g. 15e12.")
  budget.add_argument("--dataset_tokens", type=float,
                      help="Tokens in one epoch of the dataset; with --epochs.")
  budget.add_argument("--days", type=float, help="Wall-clock days available.")
  budget.add_argument("--gpu_hours", type=float, help="GPU-hours available.")
  parser.add_argument("--epochs", type=float, default=1.0)
  parser.add_argument("--attention", default="causal", choices=list(models.ATTENTION),
                      help="How the attention scores count: causal half (JAX-Titan), the "
                      "full square (PaLM, Megatron) or none (6N, as in published compute "
                      "figures).")
  parser.add_argument("--gpu", default="MI355X",
                      help="One GPU or a comma-separated list: " + ", ".join(gpus_lib.GPUS)
                      + " (MI450 is the MI455X).")
  parser.add_argument("--gpus", type=int, default=8)
  parser.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "fp4", "mxfp4"],
                      help="Which dense peak MFU is against.")
  parser.add_argument("--seq_len", type=int, default=8192)
  parser.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--mfu", type=float, default=0.40, help="Assumed sustained MFU.")
  parser.add_argument("--from_run", default="",
                      help="A JAX-Titan --run_dir: its measured throughput, power, checkpoint "
                      "and restart costs and memory overhead replace the defaults.")
  parser.add_argument("--ettr", type=float, default=None,
                      help="Default: projected for this many GPUs from the next three.")
  parser.add_argument("--w_cp_s", type=float, default=RECIPE_W_CP_S)
  parser.add_argument("--restart_s", type=float, default=RECIPE_RESTART_S)
  parser.add_argument("--failures_per_1000_gpu_days", type=float,
                      default=LLAMA3_FAILURES_PER_1000_GPU_DAYS)
  parser.add_argument("--power_fraction", type=float, default=1.0,
                      help="Mean GPU power over board power; 1 is an upper bound.")
  parser.add_argument("--pue", type=float, default=1.0)
  parser.add_argument("--usd_per_gpu_hour", type=float, default=None)
  parser.add_argument("--tp", type=int, default=1)
  parser.add_argument("--pp", type=int, default=1)
  parser.add_argument("--ep", type=int, default=1)
  parser.add_argument("--fsdp", action="store_true")
  parser.add_argument("--batch_tokens", type=float, default=4 * 2**20,
                      help="Global tokens per step, for activation memory and communication.")
  parser.add_argument("--microbatches", type=int, default=0,
                      help="Default: gradient accumulation down to one sequence per GPU "
                      "per microbatch.")
  parser.add_argument("--memory_overhead", type=float, default=None,
                      help="Peak HBM over the analytic floor. Default: measured on the "
                      f"--from_run run, else {MEASURED_MEMORY_OVERHEAD} (JAX-Titan's Llama "
                      "3.1 8B pipeline).")
  parser.add_argument("--grad_reduce_bytes", type=int, default=4,
                      help="Bytes per gradient element reduced across data parallelism; 4 "
                      "(fp32) in JAX-Titan.")
  parser.add_argument("--scaleout_gbs", type=float, default=None,
                      help="Scale-out GB/s per GPU per direction; default: the GPU table's "
                      "(one 400 Gb/s NIC per GPU for MI300-series and MI350-series).")
  parser.add_argument("--comm_efficiency", type=float, default=comms.EFFICIENCY,
                      help="Fraction of the links' peak collectives reach.")
  parser.add_argument("--state_bytes", default="4,4,8",
                      help="Bytes per parameter of weights, gradients and optimizer state: "
                      "4,4,8 is JAX-Titan's fp32 AdamW; Megatron's mixed precision is "
                      "2,4,12; bf16 moments are 4,4,4.")
  parser.add_argument("--json", default="")
  args = parser.parse_args(argv)

  model = models.from_name(args.model)
  run = from_run(args.from_run) if args.from_run else None
  results = [plan(model, gpus_lib.get(name), args, run) for name in args.gpu.split(",")]
  if len(results) == 1:
    report(results[0], args)
  else:
    comparison(results)
  if args.json:
    with open(args.json, "w") as f:
      json.dump(results[0] if len(results) == 1 else results, f, indent=1)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
