"""Measured ceilings of one GPU, for MFU against what the GPU attains.

  python -m jaxtitan.observability.ceilings OUT.json [--device 0]

Times square BF16 and FP8 matmuls through XLA (hipBLASLt) and an HBM stream
(one read and one write of a 4 GiB array), each the best of several runs, and
writes `bf16_tflops`, `fp8_tflops` and `hbm_gbs` with the device kind and
the shapes. The ROCm scaling book's Chapter 5 calibrates the same ceilings
with hipBLASLt and a streaming kernel; these are XLA's attainable numbers.
`summarize --ceilings OUT.json` reports MFU against `bf16_tflops` next to
MFU against the nominal peak.
"""

import argparse
import json
import pathlib
import time


def _best_time(fn, *args, repeats: int = 20) -> float:
  import jax  # pylint: disable=g-import-not-at-top

  jax.block_until_ready(fn(*args))
  best = float("inf")
  for _ in range(3):
    t0 = time.perf_counter()
    for _ in range(repeats):
      out = fn(*args)
    jax.block_until_ready(out)
    best = min(best, (time.perf_counter() - t0) / repeats)
  return best


def measure(device_index: int = 0, sizes=(8192, 16384)) -> dict:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  device = jax.devices()[device_index]
  out = {"device_kind": device.device_kind, "device": int(device.id), "gemm": {}}
  key = jax.random.key(0)
  for dtype, name in ((jnp.bfloat16, "bf16"), (jnp.float8_e4m3fn, "fp8")):
    best = 0.0
    for n in sizes:
      a = jax.device_put(jax.random.normal(key, (n, n), jnp.float32).astype(dtype), device)
      b = jax.device_put(jax.random.normal(key, (n, n), jnp.float32).astype(dtype), device)
      f = jax.jit(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.bfloat16))
      try:
        t = _best_time(f, a, b)
      except Exception as e:  # pylint: disable=broad-except
        out["gemm"][f"{name}_{n}"] = f"failed: {type(e).__name__}"
        continue
      tflops = 2 * n**3 / t / 1e12
      out["gemm"][f"{name}_{n}"] = tflops
      best = max(best, tflops)
      del a, b
    out[f"{name}_tflops"] = best or None

  n = 2**30
  x = jax.device_put(jnp.ones((n,), jnp.float32), device)
  stream = jax.jit(lambda v: v * 1.0001 + 1.0)
  t = _best_time(stream, x)
  out["hbm_gbs"] = 2 * 4 * n / t / 1e9
  out["hbm_bytes"] = 2 * 4 * n
  return out


def load(path) -> dict | None:
  return json.loads(pathlib.Path(path).read_text()) if path else None


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("out")
  parser.add_argument("--device", type=int, default=0)
  args = parser.parse_args(argv)
  result = measure(args.device)
  pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
  print(f"{result['device_kind']}: BF16 {result['bf16_tflops']:.0f} TFLOP/s, "
        f"FP8 {result['fp8_tflops'] or 0:.0f} TFLOP/s, HBM {result['hbm_gbs']:.0f} GB/s")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
