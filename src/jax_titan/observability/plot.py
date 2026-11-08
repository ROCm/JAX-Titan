"""Figures of training runs: MFU and step time over wall time, loss against tokens.

  python -m jax_titan.observability.plot RUN [RUN ...] [--out DIR] [--window 50]
      [--format svg,png]

RUN is a trainer `--run_dir` or log file. For each run, `<name>.svg` has:
  1. MFU of each step and its rolling MFU over `--window` steps (model FLOPs over the
     window's elapsed time) against wall time since the launch, with checkpoint saves,
     validation passes and restarts marked;
  2. step time against wall time;
  3. training loss and validation loss against tokens trained;
  4. the gradient norm against the step;
  5. GPU power and GFX clock against wall time, when the run has `power.jsonl`.
With several runs, `comparison.svg` puts their losses against tokens and their rolling
MFU against wall time side by side. Needs matplotlib (the `plots` extra); the trainer
writes `<run_dir>/figures/<run_dir name>.svg` at the end when it is installed.
"""

import argparse
import pathlib

from jax_titan.observability import records as rec
from jax_titan.observability import summarize

_NAN = float("nan")  # breaks a plotted line


def _pyplot():
  import matplotlib  # pylint: disable=g-import-not-at-top

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  return plt


def series(path, window: int = summarize.ROLLING_STEPS) -> dict:
  """Per-step and event series of one run, times in minutes since its first launch."""
  path = pathlib.Path(path)
  records = rec.read(summarize._logs(path)[0])  # pylint: disable=protected-access
  start = next(iter(rec.events(records, "start")), None)
  steps = rec.steps(records)
  if start:
    origin = summarize.origin(start)
  else:
    origin = steps[0]["time"] - steps[0]["step_time_s"] if steps and "time" in steps[0] else 0.0
  layout = next(iter(rec.events(records, "layout")), None) or {}
  tokens = layout.get("tokens_per_step") or (
      steps[0]["tokens_per_s"] * steps[0]["step_time_s"] if steps else 0)

  def minutes(t):
    return (t - origin) / 60

  rolling = []
  for attempt in rec.attempts(records):
    s = rec.steps(attempt)[1:]
    if rolling:
      rolling.append((_NAN, _NAN))
    for i in range(window, len(s) + 1):
      chunk = s[i - window:i]
      seconds = sum(r["step_time_s"] for r in chunk)
      rolling.append((minutes(chunk[-1]["time"]),
                      sum(r["mfu"] * r["step_time_s"] for r in chunk) / seconds))
  power = []
  power_file = path / "power.jsonl" if path.is_dir() else None
  if power_file and power_file.exists():
    from jax_titan.observability import power as power_lib  # pylint: disable=g-import-not-at-top

    by_time = {}
    for s in power_lib.read(power_file):
      if isinstance(s.get("power_w"), (int, float)):
        by_time.setdefault(round(s["time"]), []).append(s)
    previous = None
    for t in sorted(by_time):
      if previous is not None and t - previous > 30:
        power.append((_NAN, _NAN, _NAN))
      previous = t
      group = by_time[t]
      clocks = [s["gfx_mhz"] for s in group if isinstance(s.get("gfx_mhz"), (int, float))]
      power.append((minutes(t), sum(s["power_w"] for s in group),
                    sum(clocks) / len(clocks) if clocks else None))
  return {
      "name": path.name if path.is_dir() else path.stem,
      "time": [minutes(r["time"]) for r in steps if "time" in r],
      "step": [r["step"] for r in steps if "time" in r],
      "mfu": [r["mfu"] for r in steps if "time" in r],
      "step_time": [r["step_time_s"] for r in steps if "time" in r],
      "loss": [r["loss"] for r in steps],
      "loss_tokens": [(r["step"] + 1) * tokens for r in steps],
      "grad_norm": [r.get("grad_norm") for r in steps],
      "all_steps": [r["step"] for r in steps],
      "rolling": rolling,
      "saves": [minutes(e["time"]) for e in rec.events(records, "save")],
      "restarts": [minutes(summarize.origin(e)) for e in rec.events(records, "start")[1:]],
      "evals": [(e["step"] * tokens, e["loss"], minutes(e["time"]))
                for e in rec.events(records, "eval")],
      "power": power,
      "window": window,
  }


def _mark(ax, s, label=True):
  for i, t in enumerate(s["saves"]):
    ax.axvline(t, color="tab:gray", linestyle="--", linewidth=0.8,
               label="checkpoint save" if label and i == 0 else None)
  for i, (_, _, t) in enumerate(s["evals"]):
    ax.axvline(t, color="tab:green", linestyle=":", linewidth=0.8,
               label="validation" if label and i == 0 else None)
  for i, t in enumerate(s["restarts"]):
    ax.axvline(t, color="tab:red", linewidth=1.2, label="restart" if label and i == 0 else None)


def plot_run(s: dict, out: pathlib.Path, formats=("svg",)) -> list[pathlib.Path]:
  plt = _pyplot()
  panels = 5 if s["power"] else 4
  fig, axes = plt.subplots(panels, 1, figsize=(10, 3 * panels), constrained_layout=True)
  ax = axes[0]
  ax.plot(s["time"], [100 * m for m in s["mfu"]], ".", markersize=2, color="tab:blue",
          alpha=0.4, label="step MFU")
  if s["rolling"]:
    ax.plot([t for t, _ in s["rolling"]], [100 * m for _, m in s["rolling"]], color="tab:blue",
            label=f"rolling MFU, {s['window']} steps")
  _mark(ax, s)
  ax.set(ylabel="MFU (%)", title=f"{s['name']}: MFU over wall time")
  ax.set_ylim(bottom=0)
  ax.legend(loc="lower right", fontsize=8)
  ax = axes[1]
  ax.plot(s["time"], s["step_time"], ".", markersize=2, color="tab:orange")
  _mark(ax, s, label=False)
  ax.set(ylabel="step time (s)", title="Step time")
  if s["step_time"]:
    # Keep the steady steps readable; first steps of an attempt go off scale.
    top = 1.5 * sorted(s["step_time"])[int(0.95 * (len(s["step_time"]) - 1))]
    over = [(t, v) for t, v in zip(s["time"], s["step_time"]) if v > top]
    ax.set_ylim(0, top)
    for t, v in over:
      ax.plot(t, top, "^", color="tab:red", markersize=5, clip_on=False)
      ax.annotate(f"{v:.1f} s", (t, top), textcoords="offset points", xytext=(4, -10),
                  fontsize=7, color="tab:red")
  ax = axes[2]
  ax.plot([t / 1e6 for t in s["loss_tokens"]], s["loss"], linewidth=0.8, color="tab:purple",
          label="training loss")
  if s["evals"]:
    ax.plot([t / 1e6 for t, _, _ in s["evals"]], [l for _, l, _ in s["evals"]], "o-",
            color="tab:green", label="validation loss")
  ax.set(xlabel="tokens trained (millions)", ylabel="loss", title="Loss against tokens")
  ax.legend(fontsize=8)
  ax = axes[3]
  norms = [(st, g) for st, g in zip(s["all_steps"], s["grad_norm"]) if g]
  if norms:
    ax.semilogy([st for st, _ in norms], [g for _, g in norms], linewidth=0.8, color="tab:brown")
  ax.set(xlabel="step", ylabel="gradient norm", title="Gradient norm (before clipping)")
  if s["power"]:
    ax = axes[4]
    ax.plot([t for t, _, _ in s["power"]], [w / 1e3 for _, w, _ in s["power"]],
            color="tab:red", linewidth=0.8)
    ax.set(xlabel="wall time (minutes)", ylabel="GPU power (kW)", title="Power and GFX clock")
    clock = ax.twinx()
    clock.plot([t for t, _, c in s["power"] if c], [c for _, _, c in s["power"] if c],
               color="tab:gray", linewidth=0.8)
    clock.set_ylabel("mean GFX clock (MHz)")
  for a in (axes[0], axes[1]):
    a.set_xlabel("wall time (minutes)")
  written = []
  for fmt in formats:
    target = out / f"{s['name']}.{fmt}"
    fig.savefig(target, dpi=120)
    written.append(target)
  plt.close(fig)
  return written


def plot_comparison(runs: list[dict], out: pathlib.Path, formats=("svg",)) -> list[pathlib.Path]:
  plt = _pyplot()
  fig, (loss_ax, mfu_ax) = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
  for s in runs:
    line, = loss_ax.plot([t / 1e6 for t in s["loss_tokens"]], s["loss"], linewidth=0.8,
                         label=s["name"])
    if s["evals"]:
      loss_ax.plot([t / 1e6 for t, _, _ in s["evals"]], [l for _, l, _ in s["evals"]], "o",
                   color=line.get_color())
    if s["rolling"]:
      mfu_ax.plot([t for t, _ in s["rolling"]], [100 * m for _, m in s["rolling"]],
                  label=s["name"])
  loss_ax.set(xlabel="tokens trained (millions)", ylabel="loss",
              title="Training loss (lines) and validation loss (points)")
  mfu_ax.set(xlabel="wall time (minutes)", ylabel="rolling MFU (%)", title="Rolling MFU")
  loss_ax.legend(fontsize=8)
  mfu_ax.legend(fontsize=8)
  written = []
  for fmt in formats:
    target = out / f"comparison.{fmt}"
    fig.savefig(target, dpi=120)
    written.append(target)
  plt.close(fig)
  return written


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("runs", nargs="+")
  parser.add_argument("--out", default="", help="Default: <run_dir>/figures, or here.")
  parser.add_argument("--window", type=int, default=summarize.ROLLING_STEPS)
  parser.add_argument("--format", default="svg", help="Comma-separated: svg, png, pdf.")
  args = parser.parse_args(argv)
  formats = tuple(args.format.split(","))
  runs = [series(r, args.window) for r in args.runs]
  for path, s in zip(args.runs, runs):
    out = pathlib.Path(args.out) if args.out else (
        pathlib.Path(path) / "figures" if pathlib.Path(path).is_dir() else pathlib.Path("."))
    out.mkdir(parents=True, exist_ok=True)
    for target in plot_run(s, out, formats):
      print(f"wrote {target}")
  if len(runs) > 1:
    out = pathlib.Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    for target in plot_comparison(runs, out, formats):
      print(f"wrote {target}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
