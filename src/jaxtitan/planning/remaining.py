"""Time left in a training run, from its records so far.

  python -m jaxtitan.planning.remaining runs/a --target_tokens 20e12 [--window 200]

Time left = tokens left / (recent tokens per second while stepping x the share of the
wall time spent stepping so far). The recent throughput is over the last `--window`
steps; the share is the committed steps' time over the wall time since the first
launch, so it carries startup, saves, restarts and lost work (ETTR so far).
Rerun it as the job runs: when the estimate moves, the run's records (MFU and step
time over time, saves, restarts, power) and its RCCL and MORI telemetry say why.
"""

import argparse
import json
import pathlib
import time

from jaxtitan.observability import records


def estimate(run, target_tokens: float, window: int = 200, tokens_done: float | None = None,
             now: float | None = None) -> dict:
  path = pathlib.Path(run)
  log = records.read(path / "train.jsonl" if path.is_dir() else path)
  steps = records.steps(log)
  if not steps:
    raise ValueError(f"{run} has no step records yet")
  layout = records.events(log, "layout")
  tokens_per_step = layout[0]["tokens_per_step"] if layout else steps[-1]["tokens_per_s"] * \
      steps[-1]["step_time_s"]
  committed = len({r["step"] for r in steps})
  done = tokens_done if tokens_done is not None else committed * tokens_per_step
  recent = steps[-window:]
  tokens_per_s = len(recent) * tokens_per_step / sum(r["step_time_s"] for r in recent)
  starts = records.events(log, "start")
  first = min((next(s[k] for k in ("launch_time", "process_start", "time")
                    if s.get(k) is not None) for s in starts),
              default=steps[0]["time"] - steps[0]["step_time_s"])
  now = now or time.time()
  last = steps[-1]["time"]
  wall = max(last - first, 1e-9)
  # Committed steps only: steps redone after a restart are lost work.
  stepping = committed * sum(r["step_time_s"] for r in steps) / len(steps)
  productive = min(1.0, stepping / wall)
  left = max(0.0, target_tokens - done)
  remaining_s = left / (tokens_per_s * productive)
  return {"run": str(run), "target_tokens": target_tokens, "tokens_done": done,
          "fraction_done": done / target_tokens, "tokens_per_step": tokens_per_step,
          "recent_tokens_per_s": tokens_per_s, "recent_mfu": (
              sum(r["mfu"] for r in recent if "mfu" in r) / len(recent)
              if all("mfu" in r for r in recent) else None),
          "productive_share": productive, "remaining_s": remaining_s,
          "remaining_days": remaining_s / 86400, "eta": last + remaining_s,
          "last_record_age_s": now - last, "attempts": len(starts)}


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("run")
  parser.add_argument("--target_tokens", type=float, required=True)
  parser.add_argument("--window", type=int, default=200)
  parser.add_argument("--tokens_done", type=float, default=None,
                      help="Tokens trained before this run directory, e.g. by an earlier job.")
  parser.add_argument("--plan_days", type=float, default=None,
                      help="The planned duration, to compare with.")
  parser.add_argument("--json", default="")
  args = parser.parse_args(argv)
  r = estimate(args.run, args.target_tokens, args.window, args.tokens_done)
  mfu = f", MFU {100 * r['recent_mfu']:.1f}%" if r["recent_mfu"] is not None else ""
  print(f"{r['tokens_done'] / 1e9:,.2f}B of {r['target_tokens'] / 1e9:,.2f}B tokens "
        f"({100 * r['fraction_done']:.2f}%) over {r['attempts']} attempt(s)")
  print(f"recent {r['recent_tokens_per_s']:,.0f} tokens/s{mfu}, stepping "
        f"{100 * r['productive_share']:.1f}% of the wall time so far")
  print(f"{r['remaining_days']:.2f} days left: done about "
        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['eta']))}"
        + (f" (planned {args.plan_days:.1f} days)" if args.plan_days else ""))
  if r["last_record_age_s"] > 600:
    print(f"the last step record is {r['last_record_age_s'] / 3600:.1f} h old: is the job "
          "running?")
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(r, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
