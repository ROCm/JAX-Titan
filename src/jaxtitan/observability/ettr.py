"""ETTR, effective throughput, and the checkpoint interval at scale.

  python -m jaxtitan.observability.ettr RUN [--queue_s Q] [--json ettr_model.json]
      [--nodes 1,2,4,8,128,512,2048] [--failure_rates 2.5,3.8,5,17.5]

RUN is a trainer `--run_dir` or log file; restarts append attempts to the same
log. Following Meta FAIR's definition (arXiv 2410.21680), ETTR is productive
time over available wall time, where productive time is the committed steps
times the steady step time. The rest splits into restart overhead `u0` (each
attempt's startup), checkpoint cost `w_cp`, lost work (steps after an
attempt's last save, redone by the next attempt), validation passes and
queue time (`--queue_s`, zero on a dedicated allocation). Effective training
throughput is the steady tokens/s times ETTR: committed tokens over wall time.

The projection takes the measured `w_cp`, restore time and `u0`, and failure
rates per 1,000 node-days (Meta FAIR's 2.5-17.5; Llama 3's 419 interruptions
in 54 days on 16k GPUs, about 3.8), so MTTF(N) = 1000 / (rate * N) days. The
checkpoint interval is Daly's optimum for checkpoint cost d and MTTF M,
sqrt(2dM) [1 + sqrt(d/2M)/3 + d/18M] - d (Young's sqrt(2dM) to first order),
and the expected ETTR is (tau / (tau + d)) (1 - (R + tau/2) / M) for restart
cost R = u0 (with restore). It assumes the measured step time holds at scale.
"""

import argparse
import json
import math
import pathlib
import statistics

from jaxtitan.observability import records as rec
from jaxtitan.observability import summarize

_DAY = 86400.0


def measure(path, *, skip: int = 5, wcp_steps: int = 2, queue_s: float = 0.0) -> dict:
  path = pathlib.Path(path)
  records = rec.read(summarize._logs(path)[0])  # pylint: disable=protected-access
  attempts = rec.attempts(records)
  steady = summarize.steady_steps(records, skip, wcp_steps)
  if not steady:
    raise ValueError(f"{path} has no steady steps past --skip={skip}")
  step_s = statistics.median(r["step_time_s"] for r in steady)
  layout = next(iter(rec.events(records, "layout")), None)
  tokens_per_step = (layout or {}).get("tokens_per_step") or statistics.median(
      r["tokens_per_s"] * r["step_time_s"] for r in steady)

  committed, lost_steps, lost_s = {}, 0, 0.0
  for i, attempt in enumerate(attempts):
    steps = rec.steps(attempt)
    saves = [e["step"] for e in rec.events(attempt, "save")]
    end = next(iter(rec.events(attempt, "end")), None)
    finished = end is not None and end.get("status") == "completed"
    last_attempt = i == len(attempts) - 1
    keep_until = steps[-1]["step"] + 1 if steps and (finished or last_attempt) else max(saves, default=0)
    for r in steps:
      if r["step"] < keep_until:
        committed[r["step"]] = r
      else:
        lost_steps += 1
        lost_s += r["step_time_s"]

  def began(attempt):
    start = next(iter(rec.events(attempt, "start")), None)
    if start:
      return summarize.origin(start)
    first = attempt[0]
    return first["time"] - first.get("step_time_s", 0.0)

  first_time = began(attempts[0])
  last_time = max(r["time"] for r in records if "time" in r)
  wall = last_time - first_time + queue_s
  productive = len(committed) * step_s
  u0 = [summarize.startup(a, step_s) for a in attempts]
  saves = summarize.checkpoint_costs(records, step_s, wcp_steps)
  eval_s = sum(e["seconds"] for e in rec.events(records, "eval"))
  restores = [e["seconds"] for e in rec.events(records, "state") if e.get("restored_step")]
  losses = {
      "u0_s": sum(s.get("total", 0.0) for s in u0),
      "w_cp_s": sum(s["w_cp_s"] for s in saves),
      "lost_work_s": lost_s,
      "eval_s": eval_s,
      "queue_s": queue_s,
  }
  losses["other_s"] = wall - productive - sum(losses.values())
  ettr = productive / wall if wall > 0 else None
  start = next(iter(rec.events(records, "start")), None) or {}
  flops = (layout or {}).get("flops_per_step")
  peak = (start.get("args") or {}).get("peak_tflops_per_device")
  capacity = start.get("num_devices", 0) * (peak or 0) * 1e12
  mfu = {}
  if flops and capacity and wall > 0:
    mfu = {"mfu_steady": flops / step_s / capacity,
           "mfu_end_to_end": flops * len(committed) / wall / capacity}
  return {
      **mfu,
      "run": str(path),
      "attempts": len(attempts),
      "steady_step_s": step_s,
      "tokens_per_step": tokens_per_step,
      "committed_steps": len(committed),
      "lost_steps": lost_steps,
      "wall_s": wall,
      "productive_s": productive,
      "losses": losses,
      "ettr": ettr,
      "steady_tokens_per_s": tokens_per_step / step_s,
      "effective_tokens_per_s": len(committed) * tokens_per_step / wall if wall > 0 else None,
      "u0_stages_s": u0,
      "w_cp_s": statistics.median(s["w_cp_s"] for s in saves) if saves else None,
      "restore_s": statistics.median(restores) if restores else None,
  }


def daly_interval(w_cp: float, mttf: float) -> float:
  """Daly's optimal compute time between checkpoints (seconds)."""
  if w_cp >= 2 * mttf:
    return mttf
  r = w_cp / (2 * mttf)
  return math.sqrt(2 * w_cp * mttf) * (1 + math.sqrt(r) / 3 + r / 9) - w_cp


def expected_ettr(w_cp: float, restart_s: float, mttf: float, interval: float) -> float:
  return max(0.0, interval / (interval + w_cp) * (1 - (restart_s + interval / 2) / mttf))


def project(w_cp: float, restart_s: float, nodes, failure_rates) -> list[dict]:
  out = []
  for rate in failure_rates:
    for n in nodes:
      mttf = 1000.0 / (rate * n) * _DAY
      tau = daly_interval(w_cp, mttf)
      out.append({
          "nodes": n, "failures_per_1000_node_days": rate, "mttf_h": mttf / 3600,
          "checkpoint_interval_h": tau / 3600,
          "young_interval_h": math.sqrt(2 * w_cp * mttf) / 3600,
          "ettr": expected_ettr(w_cp, restart_s, mttf, tau),
      })
  return out


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("run")
  parser.add_argument("--skip", type=int, default=5)
  parser.add_argument("--wcp_steps", type=int, default=2)
  parser.add_argument("--queue_s", type=float, default=0.0)
  parser.add_argument("--nodes", default="1,2,4,8,16,32,64,128,256,512,1024,2048")
  parser.add_argument("--failure_rates", default="2.5,3.8,5,17.5",
                      help="Failures per 1,000 node-days.")
  parser.add_argument("--w_cp_s", type=float, default=None,
                      help="Checkpoint cost for the projection; default the run's median.")
  parser.add_argument("--restart_s", type=float, default=None,
                      help="Restart cost for the projection; default the run's largest u0.")
  parser.add_argument("--json", default="", help="Write ettr_model.json here.")
  args = parser.parse_args(argv)

  m = measure(args.run, skip=args.skip, wcp_steps=args.wcp_steps, queue_s=args.queue_s)
  print(f"{m['run']}: {m['attempts']} attempt(s), {m['committed_steps']} committed steps "
        f"({m['lost_steps']} lost), wall {m['wall_s']:.0f} s, productive {m['productive_s']:.0f} s")
  print(f"ETTR {m['ettr']:.3f}; effective {m['effective_tokens_per_s']:.0f} tokens/s against "
        f"{m['steady_tokens_per_s']:.0f} steady"
        + (f"; MFU {100 * m['mfu_end_to_end']:.2f}% end to end against "
           f"{100 * m['mfu_steady']:.2f}% steady" if "mfu_end_to_end" in m else ""))
  print("losses: " + ", ".join(f"{k[:-2]} {v:.0f} s" for k, v in m["losses"].items()))
  model = {"measured": m}
  w_cp = args.w_cp_s if args.w_cp_s is not None else m["w_cp_s"]
  restart = args.restart_s if args.restart_s is not None else max(
      (s.get("total", 0.0) for s in m["u0_stages_s"]), default=0.0)
  if w_cp:
    nodes = [int(n) for n in args.nodes.split(",")]
    rates = [float(r) for r in args.failure_rates.split(",")]
    model["projection"] = {"w_cp_s": w_cp, "restart_s": restart,
                           "rows": project(w_cp, restart, nodes, rates)}
    print(f"projection with w_cp {w_cp:.1f} s and restart {restart:.0f} s:")
    print("  nodes  failures/1000 node-days  MTTF h  interval h  ETTR")
    for row in model["projection"]["rows"]:
      print(f"  {row['nodes']:5d}  {row['failures_per_1000_node_days']:23.1f}  {row['mttf_h']:6.1f}"
            f"  {row['checkpoint_interval_h']:10.2f}  {row['ettr']:.3f}")
  else:
    print("no checkpoint saves: pass --w_cp_s to project")
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(model, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
