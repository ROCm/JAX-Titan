"""Training-run metrics, after the ROCm scaling book's A0 definitions.

  python -m jax_titan.observability.summarize RUN [RUN ...] [--skip 5]
      [--ceilings ceilings.json] [--target_loss L] [--json OUT]
  python -m jax_titan.observability.summarize --gate0 REFERENCE RUN [RUN ...]
      [--gate0_steps K] [--gate0_rtol R]

RUN is a trainer `--run_dir` (`train*.jsonl`, and `power.jsonl`,
`profile.json`, `rccl_inspector/` and `env.json` when present) or a trainer
log file. For each run (written to `<run_dir>/results.json`):

- step time p50, p95 and max over steady steps: each attempt's first `--skip`
  steps and the steps a checkpoint save slows are left out;
- tokens/s, tokens/s per GPU, and MFU against the nominal peak and, with
  `--ceilings` (`jax_titan.observability.ceilings`), the calibrated one. MFU is
  model FLOPs over elapsed time, never a mean of per-step MFUs: at the median
  step; sustained (over the steady steps); training-active (over every step);
  end to end (committed steps over the wall time of all attempts, ETTR x MFU);
  and its range over windows of 50 steps. HFU, sustained, counts the FLOPs
  rematerialization recomputes too (`jax_titan.observability.flops`);
- loss, gradient norm, non-finite steps, and peak HBM per device;
- startup stages (the restart overhead `u0`): launch, Python imports, JAX
  init, setup, state init or restore, compile, data, and the first step's
  excess over a steady step;
- checkpoints: blocking time per save and `w_cp` (blocking time plus the
  slowdown of the next `--wcp_steps` steps), and restore time;
- validation losses, and the time to `--target_loss`;
- energy per token, mean power, clocks and throttling from `power.jsonl`;
- collective completion time p50 and p99 per collective from the RCCL
  Inspector (`jax_titan.observability.collectives`), and exposed
  communication per GPU from `profile.json` (`jax_titan.observability.profile`);
- step-time spread across processes, when each process wrote its own log.

Across runs it reports repeats of one configuration (median, spread, and a
regression threshold from their variance), scaling efficiency
`E(N) = T(N) / ((N / N0) T(N0))`, the Pareto front of tokens/s per GPU
against peak HBM, and the pipeline's fixed per-step cost fitted over runs
that differ only in microbatches (pipeline utilization = 1 - fixed / step).
`--gate0` checks each run's losses over the first K steps against a
reference run, within a relative tolerance.
"""

import argparse
import json
import math
import pathlib
import statistics

from jax_titan.observability import records as rec

_CONFIG_KEYS = (
    "model", "model_overrides", "parallel", "num_stages", "schedule", "stage_layers",
    "stages_per_mesh", "mpmd_options", "reduce_grads", "per_device", "stage_fsdp",
    "tensor_parallel", "microbatches", "microbatch_size", "seq_len", "resolution", "ops",
    "remat", "max_grad_norm", "lr", "warmup_steps", "data",
)


def _quantile(values, q: float):
  if not values:
    return None
  values = sorted(values)
  pos = (len(values) - 1) * q
  lo, hi = math.floor(pos), math.ceil(pos)
  return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _stats(values) -> dict | None:
  if not values:
    return None
  return {
      "p50": _quantile(values, 0.5), "p95": _quantile(values, 0.95), "max": max(values),
      "min": min(values), "mean": statistics.fmean(values),
      "std": statistics.pstdev(values) if len(values) > 1 else 0.0, "n": len(values),
  }


def _logs(path: pathlib.Path) -> list[pathlib.Path]:
  if path.is_dir():
    logs = sorted(path.glob("train*.jsonl"))
    if not logs:
      raise FileNotFoundError(f"no train*.jsonl in {path}")
    return logs
  return [path]


def _first(records, name):
  found = rec.events(records, name)
  return found[0] if found else None


def steady_steps(records, skip: int, wcp_steps: int) -> list[dict]:
  """Step records past each attempt's first `skip` steps, minus the
  `wcp_steps` steps after each save."""
  out = []
  for attempt in rec.attempts(records):
    after_save = 0
    seen = 0
    for r in attempt:
      if r.get("event") == "save":
        after_save = wcp_steps
      elif rec.is_step(r):
        seen += 1
        if after_save:
          after_save -= 1
        elif seen > skip:
          out.append(r)
  return out


def origin(start: dict) -> float:
  """When an attempt began: its launch, else its process start, else main()."""
  for key in ("launch_time", "process_start", "main_start"):
    if start.get(key) is not None:
      return start[key]
  raise KeyError("start event without a start time")


def startup(attempt: list[dict], steady_s: float | None) -> dict:
  """Restart-overhead stages of one attempt, in seconds."""
  start = _first(attempt, "start")
  steps = rec.steps(attempt)
  if not start or not steps:
    return {}
  began = origin(start)
  stages = {}
  if start.get("launch_time") is not None and start.get("process_start") is not None:
    stages["launch"] = max(0.0, start["process_start"] - start["launch_time"])
  if start.get("process_start") is not None:
    stages["python_imports"] = max(0.0, start["main_start"] - start["process_start"])
  stages["jax_init"] = start["jax_ready"] - start["main_start"]
  state, compiled, data = (_first(attempt, n) for n in ("state", "compile", "data"))
  if state:
    stages["setup"] = max(0.0, state["time"] - state["seconds"] - start["jax_ready"])
    stages["restore" if state.get("restored_step") else "init"] = state["seconds"]
  if compiled:
    stages["compile"] = compiled["lower_s"] + compiled["compile_s"]
  if data:
    stages["data"] = data["seconds"]
  first = steps[0]
  if steady_s:
    stages["first_step_excess"] = max(0.0, first["step_time_s"] - steady_s)
  total = first["time"] - began - (steady_s or 0.0) if "time" in first else None
  if total is not None:
    stages["other"] = max(0.0, total - sum(stages.values()))
    stages["total"] = total
  return stages


def checkpoint_costs(records, steady_s: float | None, wcp_steps: int) -> list[dict]:
  """Each save: blocking seconds, and `w_cp` with the next steps' slowdown."""
  out = []
  for attempt in rec.attempts(records):
    for i, r in enumerate(attempt):
      if r.get("event") != "save":
        continue
      later = [s for s in attempt[i + 1:] if rec.is_step(s)][:wcp_steps]
      excess = sum(max(0.0, s["step_time_s"] - steady_s) for s in later) if steady_s else 0.0
      out.append({"step": r["step"], "blocking_s": r["blocking_s"],
                  "w_cp_s": r["blocking_s"] + excess})
  return out


def _config(start: dict | None) -> dict:
  args = (start or {}).get("args") or {}
  return {k: args[k] for k in _CONFIG_KEYS if k in args}


def _process_spread(logs: list[list[dict]], skip: int, wcp_steps: int) -> dict | None:
  if len(logs) < 2:
    return None
  per_step = {}
  medians = []
  for i, records in enumerate(logs):
    steady = steady_steps(records, skip, wcp_steps)
    medians.append(statistics.median(r["step_time_s"] for r in steady) if steady else None)
    for r in steady:
      per_step.setdefault(r["step"], []).append(r["step_time_s"])
  spreads = [max(v) - min(v) for v in per_step.values() if len(v) == len(logs)]
  slowest = max(range(len(medians)), key=lambda i: medians[i] or 0.0)
  return {"processes": len(logs), "spread_s": _stats(spreads), "slowest_process": slowest,
          "median_step_s_by_process": medians}


def run_metrics(path, *, skip: int = 5, wcp_steps: int = 2, ceilings: dict | None = None,
                target_loss: float | None = None) -> dict:
  path = pathlib.Path(path)
  logs = [rec.read(p) for p in _logs(path)]
  records = logs[0]
  steps = rec.steps(records)
  start = _first(records, "start")
  layout = _first(records, "layout")
  steady = steady_steps(records, skip, wcp_steps)
  times = [r["step_time_s"] for r in steady]
  out = {"run": str(path), "config": _config(start)}
  if start:
    out.update(devices=start.get("num_devices"), processes=start.get("process_count"),
               device_kind=start.get("device_kind"), host=start.get("host"))
  out["steps"] = {"count": len(steps), "steady": len(steady), "attempts": len(rec.attempts(records))}
  step_stats = _stats(times)
  out["step_time_s"] = step_stats
  p50 = step_stats["p50"] if step_stats else None
  if steps and p50:
    tokens = (layout or {}).get("tokens_per_step") or statistics.median(
        r["tokens_per_s"] * r["step_time_s"] for r in steps)
    out["tokens_per_step"] = tokens
    out["tokens_per_s"] = tokens / p50
    if out.get("devices"):
      out["tokens_per_s_per_gpu"] = out["tokens_per_s"] / out["devices"]
    flops = (layout or {}).get("flops_per_step")
    hardware = (layout or {}).get("hardware_flops_per_step")
    peak = ((start or {}).get("args") or {}).get("peak_tflops_per_device")
    if flops and peak and out.get("devices"):
      capacity = out["devices"] * peak * 1e12
      out["mfu"] = flops / p50 / capacity
      out["peak_flops_per_s"] = capacity
      out["active_s"] = sum(r["step_time_s"] for r in steps)
      out["mfu_sustained"] = flops * len(times) / sum(times) / capacity
      out["mfu_training_active"] = (
          flops * len(steps) / sum(r["step_time_s"] for r in steps) / capacity)
      rolling = [m for _, _, m in rolling_mfu(records, flops, capacity)]
      if rolling:
        out["mfu_rolling"] = {"window_steps": ROLLING_STEPS, "min": min(rolling),
                              "max": max(rolling)}
      if hardware:
        out["hfu_sustained"] = hardware * len(times) / sum(times) / capacity
      out.update(_end_to_end_mfu(path, flops, capacity, skip, wcp_steps))
    else:
      out["mfu"] = statistics.median(r["mfu"] for r in steady)
    if flops and ceilings and ceilings.get("bf16_tflops") and out.get("devices"):
      out["mfu_calibrated"] = flops / p50 / (out["devices"] * ceilings["bf16_tflops"] * 1e12)
      out["ceilings"] = {k: ceilings.get(k) for k in ("bf16_tflops", "fp8_tflops", "hbm_gbs")}
  if steps:
    losses = [r["loss"] for r in steps]
    finite = [x for x in losses if math.isfinite(x)]
    out["loss"] = {"first": losses[0], "last": losses[-1], "min": min(finite) if finite else None}
    norms = [r["grad_norm"] for r in steps if "grad_norm" in r]
    finite_norms = [x for x in norms if math.isfinite(x)]
    if finite_norms:
      out["grad_norm"] = {"p50": statistics.median(finite_norms), "max": max(finite_norms)}
    out["non_finite_steps"] = sum(1 for x in losses if not math.isfinite(x))
  memory = _first(records, "memory")
  if memory:
    out["peak_hbm_gib"] = {"max": max(memory["peak_hbm_gib"]), "min": min(memory["peak_hbm_gib"])}
  compiled = _first(records, "compile")
  if compiled:
    out["compile_s"] = compiled["compile_s"]
  attempts = rec.attempts(records)
  out["startup_s"] = startup(attempts[0], p50)
  saves = checkpoint_costs(records, p50, wcp_steps)
  restores = [e["seconds"] for e in rec.events(records, "state") if e.get("restored_step")]
  if saves or restores:
    out["checkpoint"] = {
        "saves": len(saves),
        "blocking_s": _stats([s["blocking_s"] for s in saves]),
        "w_cp_s": _stats([s["w_cp_s"] for s in saves]),
        "restore_s": _stats(restores),
    }
  evals = rec.events(records, "eval")
  if evals:
    out["eval"] = [{"step": e["step"], "loss": e["loss"], "seconds": e["seconds"]} for e in evals]
    if target_loss is not None:
      out["time_to_target"] = time_to_target(records, target_loss)
  spread = _process_spread(logs, skip, wcp_steps)
  if spread:
    out["process_spread"] = spread
  if path.is_dir():
    out.update(_run_dir_extras(path, records, steady, out))
  return out


ROLLING_STEPS = 50


def rolling_mfu(records, flops_per_step: float, capacity: float,
                window: int = ROLLING_STEPS) -> list[tuple[float, int, float]]:
  """(wall time, step, MFU) over each window of `window` consecutive steps of an
  attempt, past its first step (one-time setup): model FLOPs over the window's
  summed step time, not a mean of step MFUs."""
  out = []
  for attempt in rec.attempts(records):
    steps = rec.steps(attempt)[1:]
    for i in range(window, len(steps) + 1):
      chunk = steps[i - window:i]
      seconds = sum(r["step_time_s"] for r in chunk)
      out.append((chunk[-1].get("time", 0.0), chunk[-1]["step"],
                  flops_per_step * window / seconds / capacity))
  return out


def _end_to_end_mfu(path, flops: float, capacity: float, skip: int, wcp_steps: int) -> dict:
  """MFU over the whole wall time of all attempts, from committed steps (ETTR x MFU)."""
  from jax_titan.observability import ettr  # pylint: disable=g-import-not-at-top

  try:
    m = ettr.measure(path, skip=skip, wcp_steps=wcp_steps)
  except (ValueError, KeyError):
    return {}
  return {"mfu_end_to_end": flops * m["committed_steps"] / m["wall_s"] / capacity,
          "ettr": m["ettr"]}


def time_to_target(records, target: float) -> dict | None:
  """Wall time from the first launch, and summed step time, until the first
  validation loss at or below `target`."""
  start = _first(records, "start")
  hit = next((e for e in rec.events(records, "eval") if e["loss"] <= target), None)
  if not hit or not start:
    return {"target": target, "reached": False}
  last = {}
  for r in rec.steps(records):
    if r["step"] < hit["step"]:
      last[r["step"]] = r["step_time_s"]
  return {"target": target, "reached": True, "step": hit["step"], "loss": hit["loss"],
          "wall_s": hit["time"] - origin(start), "train_s": sum(last.values())}


def _run_dir_extras(path: pathlib.Path, records, steady, out) -> dict:
  extras = {}
  power_file = path / "power.jsonl"
  if power_file.exists() and len(steady) >= 2 and out.get("tokens_per_step"):
    from jax_titan.observability import power  # pylint: disable=g-import-not-at-top

    t0 = steady[0]["time"] - steady[0]["step_time_s"]
    t1 = steady[-1]["time"]
    samples = power.read(power_file)
    per_gpu = power.energy_between(samples, t0, t1)
    if per_gpu:
      watts = sum(g["mean_power_w"] for g in per_gpu.values() if g["mean_power_w"])
      steps_in_window = sum(1 for r in rec.steps(records) if t0 < r["time"] <= t1)
      tokens = steps_in_window * out["tokens_per_step"]
      clocks = [g["mean_gfx_mhz"] for g in per_gpu.values() if g["mean_gfx_mhz"]] or [0.0]
      source = next((s.get("source") for s in samples if s.get("source")), None)
      extras["energy"] = {
          "boundary": "GPU",
          "source": source,
          "energy_per_token_j": watts * (t1 - t0) / tokens if tokens else None,
          "mean_power_w": watts,
          "gfx_mhz": {"min": min(clocks), "max": max(clocks)},
          "per_gpu": per_gpu,
      }
  profile_file = path / "profile.json"
  if profile_file.exists():
    profile = json.loads(profile_file.read_text())
    extras["exposed_communication"] = profile
  inspector = path / "rccl_inspector"
  if inspector.is_dir():
    from jax_titan.observability import collectives  # pylint: disable=g-import-not-at-top

    extras["collectives"] = collectives.summarize(inspector)
  return extras


def write_results(run_dir, **kwargs) -> dict:
  metrics = run_metrics(run_dir, **kwargs)
  pathlib.Path(run_dir, "results.json").write_text(json.dumps(metrics, indent=1) + "\n")
  return metrics


def _config_key(metrics: dict, drop=("seed",)) -> str:
  return json.dumps({k: v for k, v in metrics["config"].items() if k not in drop}, sort_keys=True)


def repeats(runs: list[dict]) -> list[dict]:
  groups = {}
  for m in runs:
    if m.get("step_time_s"):
      groups.setdefault(_config_key(m), []).append(m)
  out = []
  for key, ms in groups.items():
    if len(ms) < 2:
      continue
    p50s = [m["step_time_s"]["p50"] for m in ms]
    median = statistics.median(p50s)
    std = statistics.pstdev(p50s)
    out.append({"config": json.loads(key), "runs": [m["run"] for m in ms], "p50_s": p50s,
                "median_s": median, "spread": (max(p50s) - min(p50s)) / median,
                "cv": std / median, "regression_threshold_s": median + 3 * std})
  return out


def aggregate_mfu(runs: list[dict]) -> float | None:
  """Model FLOPs of every run over the peak capacity they had while stepping."""
  usable = [m for m in runs if m.get("mfu_training_active") and m.get("active_s")]
  if len(usable) < 2:
    return None
  capacity = sum(m["peak_flops_per_s"] * m["active_s"] for m in usable)
  return sum(m["mfu_training_active"] * m["peak_flops_per_s"] * m["active_s"]
             for m in usable) / capacity


def scaling(runs: list[dict]) -> list[dict]:
  points = sorted((m for m in runs if m.get("tokens_per_s") and m.get("devices")),
                  key=lambda m: m["devices"])
  if len(points) < 2:
    return []
  base = points[0]
  out = []
  for m in points:
    ratio = m["devices"] / base["devices"]
    out.append({"run": m["run"], "devices": m["devices"], "tokens_per_s": m["tokens_per_s"],
                "tokens_per_step": m.get("tokens_per_step"),
                "efficiency": m["tokens_per_s"] / (ratio * base["tokens_per_s"])})
  return out


def pareto(runs: list[dict]) -> list[dict]:
  points = [m for m in runs if m.get("tokens_per_s_per_gpu") and m.get("peak_hbm_gib")]
  front = []
  for m in points:
    dominated = any(
        o is not m and o["tokens_per_s_per_gpu"] >= m["tokens_per_s_per_gpu"]
        and o["peak_hbm_gib"]["max"] <= m["peak_hbm_gib"]["max"]
        and (o["tokens_per_s_per_gpu"] > m["tokens_per_s_per_gpu"]
             or o["peak_hbm_gib"]["max"] < m["peak_hbm_gib"]["max"])
        for o in points)
    if not dominated:
      front.append({"run": m["run"], "tokens_per_s_per_gpu": m["tokens_per_s_per_gpu"],
                    "peak_hbm_gib": m["peak_hbm_gib"]["max"], "config": m["config"]})
  return sorted(front, key=lambda p: p["peak_hbm_gib"])


def bubble(runs: list[dict]) -> list[dict]:
  """Fixed per-step cost from step time against microbatch count."""
  groups = {}
  for m in runs:
    if m.get("step_time_s") and "microbatches" in m["config"]:
      groups.setdefault(_config_key(m, drop=("seed", "microbatches")), []).append(m)
  out = []
  for key, ms in groups.items():
    xs = [m["config"]["microbatches"] for m in ms]
    if len(set(xs)) < 2:
      continue
    ys = [m["step_time_s"]["p50"] for m in ms]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    fixed = my - slope * mx
    out.append({"config": json.loads(key), "fixed_s": fixed, "per_microbatch_s": slope,
                "utilization": {str(x): 1 - fixed / y for x, y in zip(xs, ys)}})
  return out


def gate0(reference, runs, *, steps: int = 10, rtol: float = 1e-2) -> list[dict]:
  ref = {r["step"]: r for r in rec.steps(rec.read(_logs(pathlib.Path(reference))[0]))}
  out = []
  for run in runs:
    got = {r["step"]: r for r in rec.steps(rec.read(_logs(pathlib.Path(run))[0]))}
    common = [s for s in sorted(ref) if s in got][:steps]
    rel = {key: [abs(got[s][key] - ref[s][key]) / max(abs(ref[s][key]), 1e-12)
                 for s in common if key in got[s] and key in ref[s]]
           for key in ("loss", "grad_norm")}
    worst = max(rel["loss"], default=float("inf"))
    out.append({"run": str(run), "reference": str(reference), "steps": len(common),
                "max_rel_loss": worst,
                "max_rel_grad_norm": max(rel["grad_norm"], default=None),
                "step0_rel_loss": rel["loss"][0] if rel["loss"] else None,
                "rtol": rtol, "pass": len(common) == steps and worst <= rtol})
  return out


def _fmt(x, spec=".3f"):
  return "-" if x is None else format(x, spec)


def print_run(m: dict):
  st = m.get("step_time_s") or {}
  print(f"{m['run']}")
  print(f"  steps {m['steps']['count']} ({m['steps']['steady']} steady, {m['steps']['attempts']} "
        f"attempt(s)) | step p50 {_fmt(st.get('p50'))} s  p95 {_fmt(st.get('p95'))}  "
        f"max {_fmt(st.get('max'))}")
  print(f"  tokens/s {_fmt(m.get('tokens_per_s'), '.0f')} | per GPU "
        f"{_fmt(m.get('tokens_per_s_per_gpu'), '.0f')} | MFU {_fmt(100 * m['mfu'] if m.get('mfu') else None, '.1f')}%"
        + (f" (calibrated {100 * m['mfu_calibrated']:.1f}%)" if m.get("mfu_calibrated") else ""))
  if m.get("mfu_sustained"):
    parts = [f"sustained {100 * m['mfu_sustained']:.2f}%",
             f"training-active {100 * m['mfu_training_active']:.2f}%"]
    if m.get("mfu_end_to_end"):
      parts.append(f"end to end {100 * m['mfu_end_to_end']:.2f}% (ETTR {m['ettr']:.3f})")
    if m.get("mfu_rolling"):
      r = m["mfu_rolling"]
      parts.append(f"{r['window_steps']}-step rolling {100 * r['min']:.2f}-{100 * r['max']:.2f}%")
    if m.get("hfu_sustained"):
      parts.append(f"HFU {100 * m['hfu_sustained']:.2f}%")
    print("  MFU " + ", ".join(parts))
  if m.get("loss"):
    print(f"  loss {m['loss']['first']:.4f} -> {m['loss']['last']:.4f}"
          + (f" | grad norm p50 {m['grad_norm']['p50']:.3f}" if m.get("grad_norm") else "")
          + (f" | peak HBM {m['peak_hbm_gib']['min']:.1f}-{m['peak_hbm_gib']['max']:.1f} GiB"
             if m.get("peak_hbm_gib") else ""))
  if m.get("startup_s"):
    print("  startup " + ", ".join(f"{k} {v:.1f}" for k, v in m["startup_s"].items()) + " s")
  if m.get("checkpoint"):
    c = m["checkpoint"]
    print(f"  checkpoint: {c['saves']} saves, blocking p50 {_fmt((c['blocking_s'] or {}).get('p50'), '.1f')} s, "
          f"w_cp p50 {_fmt((c['w_cp_s'] or {}).get('p50'), '.1f')} s, restore "
          f"{_fmt((c['restore_s'] or {}).get('p50'), '.1f')} s")
  if m.get("eval"):
    print("  validation loss " + ", ".join(f"{e['step']}: {e['loss']:.4f}" for e in m["eval"]))
  if m.get("time_to_target"):
    t = m["time_to_target"]
    print(f"  time to {t['target']}: " + (f"step {t['step']}, {t['wall_s']:.0f} s wall, "
                                         f"{t['train_s']:.0f} s training" if t["reached"] else "not reached"))
  if m.get("energy"):
    e = m["energy"]
    print(f"  energy {_fmt(e['energy_per_token_j'], '.3f')} J/token ({e['boundary']}, {e['source']}), "
          f"{e['mean_power_w']:.0f} W, GFX {e['gfx_mhz']['min']:.0f}-{e['gfx_mhz']['max']:.0f} MHz")
  if m.get("process_spread"):
    s = m["process_spread"]
    print(f"  across {s['processes']} processes: step spread p50 "
          f"{_fmt((s['spread_s'] or {}).get('p50'), '.3f')} s, slowest process {s['slowest_process']}")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("runs", nargs="+")
  parser.add_argument("--skip", type=int, default=5, help="Warmup steps left out of each attempt.")
  parser.add_argument("--wcp_steps", type=int, default=2,
                      help="Steps after a save whose slowdown counts toward w_cp.")
  parser.add_argument("--ceilings", default="", help="ceilings.json for calibrated MFU.")
  parser.add_argument("--target_loss", type=float, default=None)
  parser.add_argument("--json", default="", help="Write every result here.")
  parser.add_argument("--no_write", action="store_true", help="Do not write results.json.")
  parser.add_argument("--gate0", action="store_true",
                      help="Check runs 2.. against run 1's losses over --gate0_steps steps.")
  parser.add_argument("--gate0_steps", type=int, default=10)
  parser.add_argument("--gate0_rtol", type=float, default=1e-2)
  args = parser.parse_args(argv)

  if args.gate0:
    result = gate0(args.runs[0], args.runs[1:], steps=args.gate0_steps, rtol=args.gate0_rtol)
    for g in result:
      print(f"{'PASS' if g['pass'] else 'FAIL'} {g['run']}: {g['steps']} steps, max relative "
            f"loss difference {g['max_rel_loss']:.2e} (step 0 {_fmt(g['step0_rel_loss'], '.1e')}), "
            f"gradient norm {_fmt(g['max_rel_grad_norm'], '.2e')}, tolerance {g['rtol']:.0e}")
    if args.json:
      pathlib.Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return 0 if all(g["pass"] for g in result) else 1

  from jax_titan.observability import ceilings as ceilings_lib  # pylint: disable=g-import-not-at-top

  ceilings = ceilings_lib.load(args.ceilings)
  runs = []
  for run in args.runs:
    kwargs = dict(skip=args.skip, wcp_steps=args.wcp_steps, ceilings=ceilings,
                  target_loss=args.target_loss)
    path = pathlib.Path(run)
    m = write_results(path, **kwargs) if path.is_dir() and not args.no_write else run_metrics(path, **kwargs)
    runs.append(m)
    print_run(m)
  report = {"runs": runs, "repeats": repeats(runs), "scaling": scaling(runs),
            "pareto": pareto(runs), "pipeline_fit": bubble(runs),
            "aggregate_mfu": aggregate_mfu(runs)}
  if report["aggregate_mfu"] is not None:
    print(f"aggregate MFU over {len(runs)} runs: {100 * report['aggregate_mfu']:.2f}%")
  for r in report["repeats"]:
    print(f"repeats ({len(r['runs'])}): median {r['median_s']:.3f} s, spread {100 * r['spread']:.1f}%, "
          f"CV {100 * r['cv']:.1f}%, regression threshold {r['regression_threshold_s']:.3f} s")
  if len({m.get('devices') for m in runs}) > 1:
    for s in report["scaling"]:
      print(f"scaling: {s['devices']} devices, {s['tokens_per_s']:.0f} tokens/s, E(N) {s['efficiency']:.3f}")
  if len(report["pareto"]) < len(runs):
    for p in report["pareto"]:
      print(f"pareto: {p['tokens_per_s_per_gpu']:.0f} tokens/s/GPU at {p['peak_hbm_gib']:.1f} GiB: {p['run']}")
  for b in report["pipeline_fit"]:
    util = ", ".join(f"{k} microbatches {100 * v:.0f}%" for k, v in b["utilization"].items())
    print(f"pipeline fit: fixed {b['fixed_s']:.3f} s per step, {b['per_microbatch_s']:.3f} s per "
          f"microbatch; utilization {util}")
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
