"""M0/M1 spike: a two-stage MLP train step through Shardy MPMD.

Lowers the step with `mpmd.jit`, dumps the MPMD and IFRT IR, then compiles and
runs it on two disjoint meshes and checks the result against a single-mesh
`jax.jit` reference.

  python examples/mpmd_two_mesh_spike.py --platform=cpu
  python examples/mpmd_two_mesh_spike.py --platform=rocm
"""

import argparse
import collections
import os
import pathlib
import re
import sys
import time

from jaxtitan import paths


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", default="cpu", choices=["cpu", "rocm"])
  parser.add_argument("--dump_dir", default=str(paths.work_dir() / "dumps/two_mesh"))
  parser.add_argument("--steps", type=int, default=5)
  return parser.parse_args()


def main():
  args = parse_args()
  if args.platform == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top

  if args.platform == "cpu":
    jax.config.update("jax_num_cpu_devices", 8)

  import jaxtitan  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  devices = jax.devices()
  print(f"jax {jax.__version__}, {len(devices)} devices: {devices[0].device_kind}")
  assert len(devices) >= 8, "needs 8 devices"
  topology = {
      "mesh0": jax.sharding.Mesh(np.array(devices[:4]), ("x",)),
      "mesh1": jax.sharding.Mesh(np.array(devices[4:8]), ("x",)),
  }
  config = mpmd.make_config(topology, {"stage0": "mesh0", "stage1": "mesh1"})

  def stage0(w0, x):
    return jax.nn.relu(x @ w0)

  def stage1(w1, h, y):
    return jnp.mean((h @ w1 - y) ** 2)

  def loss_mpmd(params, x, y):
    h = mpmd.named_computation(stage0, name="stage0")(params["w0"], x)
    return mpmd.named_computation(stage1, name="stage1")(params["w1"], h, y)

  def loss_ref(params, x, y):
    return stage1(params["w1"], stage0(params["w0"], x), y)

  def make_step(loss_fn):
    def step(params, x, y):
      loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
      params = jax.tree.map(lambda p, g: p - 0.05 * g, params, grads)
      return params, loss

    return step

  key = jax.random.key(0)
  k0, k1, kx, ky = jax.random.split(key, 4)
  params = {
      "w0": jax.random.normal(k0, (64, 128), jnp.float32) * 0.1,
      "w1": jax.random.normal(k1, (128, 32), jnp.float32) * 0.1,
  }
  x = jax.random.normal(kx, (16, 64), jnp.float32)
  y = jax.random.normal(ky, (16, 32), jnp.float32)

  replicated = {
      name: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      for name, mesh in topology.items()
  }
  param_shardings = {"w0": replicated["mesh0"], "w1": replicated["mesh1"]}
  in_shardings = (param_shardings, replicated["mesh0"], replicated["mesh1"])
  host_params, host_x, host_y = jax.device_get((params, x, y))
  params, x, y = jax.device_put((params, x, y), in_shardings)

  t0 = time.time()
  lowered = mpmd.jit(
      make_step(loss_mpmd), config, in_shardings=in_shardings,
      out_shardings=(param_shardings, replicated["mesh1"]),
  ).lower(params, x, y)
  print(f"lowered in {time.time() - t0:.2f}s")

  dump_dir = pathlib.Path(args.dump_dir)
  dump_dir.mkdir(parents=True, exist_ok=True)
  for dialect in ("mpmd", "ifrt"):
    text = lowered.as_text(dialect)
    (dump_dir / f"train_step.{dialect}.mlir").write_text(text)
    ops = collections.Counter(re.findall(rf"\b{dialect}\.[A-Za-z_]+", text))
    print(f"{dialect} ops: {dict(ops)}")
  print(f"IR dumped to {dump_dir}")

  t0 = time.time()
  try:
    compiled = lowered.compile()
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"compile failed after {time.time() - t0:.2f}s: {type(e).__name__}: {e}")
    return 2
  print(f"compiled in {time.time() - t0:.2f}s")

  ref_step = jax.jit(make_step(loss_ref))
  ref_params = host_params
  mpmd_params = params
  for i in range(args.steps):
    mpmd_params, mpmd_loss = compiled(mpmd_params, x, y)
    ref_params, ref_loss = ref_step(ref_params, host_x, host_y)
    print(f"step {i}: mpmd loss {float(mpmd_loss):.6f}  ref loss {float(ref_loss):.6f}")
  for name in host_params:
    np.testing.assert_allclose(
        np.asarray(mpmd_params[name]), np.asarray(ref_params[name]),
        rtol=1e-5, atol=1e-5,
    )
  print("params match the single-mesh reference")
  print("w0 sharding:", mpmd_params["w0"].sharding)
  print("w1 sharding:", mpmd_params["w1"].sharding)
  return 0


if __name__ == "__main__":
  sys.exit(main())
