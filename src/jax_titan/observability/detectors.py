"""Step-time overhead of each failure detector (the ROCm scaling book, Chapter 13).

  python -m jax_titan.observability.detectors OUT_DIR [--only a,b] [--repeats N]
      -- TRAINER_ARGS...

Runs the trainer once per detector (and `--repeats` times each), every run a
`--run_dir` under OUT_DIR, and writes OUT_DIR/detector_overhead.json with
each detector's median step time against the baseline's. The baseline turns
the step watchdog off and keeps XLA's and RCCL's defaults, which already
include XLA's stuck and terminate timeouts and RCCL RAS. Detectors:

  watchdog             the trainer's step watchdog (--step_timeout=300)
  progress_tracking    XLA progress tracking of the last 16 thunks
  detect_nan_inf       XLA's NaN and Inf checks on every thunk (warning mode)
  checksum_tracing     XLA's per-thunk buffer checksums
  rccl_ras_off         RCCL RAS off, so its cost is baseline / this - 1
  debug_agent          the ROCr Debug Agent loaded (HSA_TOOLS_LIB)
  debug_agent_precise  the same with precise ALU exceptions
  rccl_inspector       the RCCL Inspector profiler plugin

A detector whose run fails, such as an XLA flag this jaxlib does not know, is
reported as failed. Run it on a free node: other jobs change step times.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import statistics
import subprocess
import sys


def _debug_agent() -> str | None:
  candidates = ["/opt/rocm/lib"]
  spec = importlib.util.find_spec("_rocm_sdk_devel")
  if spec and spec.submodule_search_locations:
    candidates.insert(0, os.path.join(spec.submodule_search_locations[0], "lib"))
  for d in candidates:
    path = os.path.join(d, "librocm-debug-agent.so.2")
    if os.path.exists(path):
      return path
  return None


def variants() -> dict:
  agent = _debug_agent()
  out = {
      "baseline": {},
      "watchdog": {"args": ["--step_timeout=300"]},
      "progress_tracking": {"xla": ["--xla_gpu_execution_progress_tracking=16"]},
      "detect_nan_inf": {"xla": ["--xla_gpu_detect_nan=warning", "--xla_gpu_detect_inf=warning"]},
      "checksum_tracing": {"xla": ["--xla_gpu_experimental_enable_checksum_tracing_on_thunks=true"]},
      "rccl_ras_off": {"env": {"NCCL_RAS_ENABLE": "0"}},
      "rccl_inspector": {"args": ["--rccl_inspector"]},
  }
  if agent:
    out["debug_agent"] = {"env": {"HSA_TOOLS_LIB": agent}}
    out["debug_agent_precise"] = {
        "env": {"HSA_TOOLS_LIB": agent, "ROCM_DEBUG_AGENT_OPTIONS": "--precise-alu-exceptions"}}
  return out


def run(out_dir: pathlib.Path, name: str, variant: dict, trainer_args, repeat: int,
        timeout: float) -> dict:
  run_dir = out_dir / (name if repeat == 0 else f"{name}.{repeat}")
  env = dict(os.environ)
  env.update(variant.get("env", {}))
  if variant.get("xla"):
    env["XLA_FLAGS"] = " ".join([env.get("XLA_FLAGS", ""), *variant["xla"]]).strip()
  args = list(variant.get("args", []))
  if not any(a.startswith("--step_timeout") for a in args):
    args.append("--step_timeout=0")
  cmd = [sys.executable, "-m", "jax_titan.trainer.train", *trainer_args, *args,
         f"--run_dir={run_dir}", "--power_sample_s=0"]
  run_dir.mkdir(parents=True, exist_ok=True)
  with open(run_dir / "stdout.txt", "w") as f:
    try:
      result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
      return {"run_dir": str(run_dir), "failed": f"timed out after {timeout:.0f} s"}
  results = run_dir / "results.json"
  if result.returncode != 0 or not results.exists():
    return {"run_dir": str(run_dir), "failed": f"exit {result.returncode}"}
  metrics = json.loads(results.read_text())
  return {"run_dir": str(run_dir), "p50_s": (metrics.get("step_time_s") or {}).get("p50")}


def main(argv=None):
  argv = sys.argv[1:] if argv is None else list(argv)
  trainer_args = []
  if "--" in argv:
    i = argv.index("--")
    argv, trainer_args = argv[:i], argv[i + 1:]
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("out_dir")
  parser.add_argument("--only", default="", help="Comma-separated detectors; baseline always runs.")
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument("--timeout", type=float, default=1800, help="Seconds per run.")
  args = parser.parse_args(argv)

  out_dir = pathlib.Path(args.out_dir)
  chosen = variants()
  if args.only:
    keep = {"baseline", *args.only.split(",")}
    chosen = {k: v for k, v in chosen.items() if k in keep}
  runs = {name: [] for name in chosen}
  for repeat in range(args.repeats):
    for name, variant in chosen.items():
      runs[name].append(run(out_dir, name, variant, trainer_args, repeat, args.timeout))
      print(f"{name} #{repeat}: {runs[name][-1]}", flush=True)

  def median(name):
    p50s = [r["p50_s"] for r in runs[name] if r.get("p50_s")]
    return statistics.median(p50s) if p50s else None

  base = median("baseline")
  table = []
  for name in chosen:
    m = median(name)
    row = {"detector": name, "p50_s": m, "runs": runs[name]}
    if m and base:
      row["overhead"] = base / m - 1 if name == "rccl_ras_off" else m / base - 1
    if m is None:
      row["failed"] = True
    table.append(row)
  (out_dir / "detector_overhead.json").write_text(
      json.dumps({"baseline_p50_s": base, "trainer_args": trainer_args, "detectors": table},
                 indent=1) + "\n")
  print("detector             step p50 s   overhead")
  for row in table:
    over = f"{100 * row['overhead']:+7.1f}%" if "overhead" in row else "   failed" if row.get("failed") else ""
    print(f"{row['detector']:20s} {row['p50_s'] or float('nan'):10.3f}   {over}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
