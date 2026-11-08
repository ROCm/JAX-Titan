"""Performance database: predicted and measured step parts, one JSON record per line.

  python -m jax_titan.planning.step --calibrate runs/a --perfdb perf.jsonl
  python -m jax_titan.observability.perfdb perf.jsonl [--threshold 0.03]

Each record keeps what a comparison needs: the model, GPU and layout (the key), the
software (package versions and commits from the run's env.json, the ROCm release and
GPU driver and firmware), the topology, the predicted and measured parts and the part
with the largest gap. The report flags a record whose measured step is slower than the
best earlier record with the same key by more than the threshold: a regression after a
change of XLA, RCCL, a kernel provider, the driver or the firmware.
"""

import argparse
import dataclasses
import json
import pathlib
import time


def _software(env: dict) -> dict:
  sw = env.get("software", {})
  gpus = env.get("hardware", {}).get("gpus") or []
  first = gpus[0] if gpus and isinstance(gpus[0], dict) else {}
  return {"packages": sw.get("packages"), "commits": sw.get("commits"), "rocm": sw.get("rocm"),
          "driver": env.get("hardware", {}).get("amdgpu_driver"),
          "vbios": (first.get("vbios") or {}).get("version"),
          "firmware": (first.get("vbios") or {}).get("boot_firmware")}


def step_record(run_dir, model: str, gpu: str, layout, calibration: dict) -> dict:
  env_path = pathlib.Path(run_dir) / "env.json"
  env = json.loads(env_path.read_text()) if env_path.exists() else {}
  hw = env.get("hardware", {})
  lo = dataclasses.asdict(layout) if dataclasses.is_dataclass(layout) else dict(layout)
  return {
      "time": time.time(), "kind": "step", "run": str(run_dir),
      "key": {"model": model, "gpu": gpu, **{k: lo[k] for k in sorted(lo)}},
      "software": _software(env),
      "topology": {"host": hw.get("host"), "gpus": lo.get("gpus"),
                   "gpus_per_node": lo.get("gpus_per_node"), "nics": hw.get("nics")},
      "predicted": {p["part"]: p["predicted_s"] for p in calibration.get("parts", [])},
      "measured": {p["part"]: p["measured_s"] for p in calibration.get("parts", [])},
      "fit": calibration.get("fit"), "bottleneck": calibration.get("bottleneck"),
  }


def append(path, record: dict) -> None:
  with open(path, "a") as f:
    f.write(json.dumps(record, default=str) + "\n")


def read(path) -> list[dict]:
  return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line]


def regressions(records: list[dict], threshold: float = 0.03) -> list[dict]:
  """Records whose measured step is slower than the best earlier one with their key."""
  best, out = {}, []
  for r in sorted(records, key=lambda r: r["time"]):
    step = (r.get("measured") or {}).get("step")
    if step is None:
      continue
    key = json.dumps(r["key"], sort_keys=True)
    if key in best and step > best[key]["measured"]["step"] * (1 + threshold):
      out.append({"record": r, "best": best[key],
                  "slower": step / best[key]["measured"]["step"] - 1})
    if key not in best or step < best[key]["measured"]["step"]:
      best[key] = r
  return out


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("path")
  parser.add_argument("--threshold", type=float, default=0.03)
  args = parser.parse_args(argv)
  records = read(args.path)
  for r in records:
    m, p = r.get("measured") or {}, r.get("predicted") or {}
    print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['time']))}  {r['key']['model']} "
          f"on {r['key']['gpus']} x {r['key']['gpu']}: step {m.get('step', float('nan')):.3f} s "
          f"measured, {p.get('step', float('nan')):.3f} s predicted, largest gap "
          f"{r.get('bottleneck')}  ({r['run']})")
  for g in regressions(records, args.threshold):
    print(f"regression: {g['record']['run']} is {100 * g['slower']:.1f}% slower than "
          f"{g['best']['run']} with the same model and layout")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
