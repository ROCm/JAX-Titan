"""`env.json`: what a run ran on, after the ROCm scaling book's A0 schema.

  python -m jaxtitan.observability.env OUT.json

Software (Python packages, git commits, ROCm), configuration (XLA flags and
the RCCL, HSA, HIP and JAX environment), hardware (host, CPUs, GPUs with PCI
bus IDs, driver, VBIOS and power caps, RDMA NICs) and, from the trainer, the
args and a rank map: each local device with its process, HIP index, PCI bus
ID and, for pipelines, its mesh and stages. Anything that cannot be read is
left out rather than failing the run.
"""

import argparse
import ctypes
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import platform
import re
import socket
import subprocess
import sys

_PACKAGES = re.compile(
    r"^(jax|jaxlib|jax[-_].*|flax|optax|orbax.*|chex|ml[-_]dtypes|numpy|maxtext|maxdiffusion|"
    r"transformer[-_]engine.*|triton|jax[-_]triton|.*aiter.*|shardy|rocm.*|amd.*|tensorflow.*)$",
    re.I,
)
_ENV = re.compile(
    r"^(XLA_|JAX_|NCCL_|RCCL_|HSA_|HIP_|ROCR_|ROCM_|GPU_|AMD_|NVTE_|MIOPEN_|HIPBLASLT_|"
    r"TRITON_|TF_CPP_|JAXTITAN_|OMP_|LD_LIBRARY_PATH$|PYTHONPATH$)"
)


def _read(path) -> str | None:
  try:
    return pathlib.Path(path).read_text().strip()
  except OSError:
    return None


def _git(path: pathlib.Path) -> dict | None:
  def run(*cmd):
    out = subprocess.run(["git", "-C", str(path), *cmd], capture_output=True, text=True,
                         timeout=10, check=False)
    return out.stdout.strip() if out.returncode == 0 else None

  commit = run("rev-parse", "HEAD")
  if not commit:
    return None
  info = {"commit": commit, "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
  submodules = run("submodule", "status")
  if submodules:
    info["submodules"] = {
        parts[1]: parts[0].lstrip("+-U") for parts in (line.split() for line in submodules.splitlines())
    }
  return info


def software() -> dict:
  from jaxtitan import paths  # pylint: disable=g-import-not-at-top

  packages = {}
  for dist in importlib.metadata.distributions():
    name = dist.metadata["Name"] or ""
    if _PACKAGES.match(name):
      packages[name] = dist.version
  repo = pathlib.Path(__file__).resolve().parents[3]
  commits = {"jax-titian": _git(repo)}
  src = paths.work_dir() / "src"
  if src.is_dir():
    for d in sorted(src.iterdir()):
      if (d / ".git").exists():
        commits[d.name] = _git(d)
  rocm = _read("/opt/rocm/.info/version") or next(
      (v for k, v in sorted(packages.items()) if k.lower().startswith("rocm-sdk")), None)
  return {
      "python": sys.version.split()[0],
      "packages": dict(sorted(packages.items())),
      "commits": {k: v for k, v in commits.items() if v},
      "rocm": rocm,
  }


def configuration(args=None) -> dict:
  env = {k: v for k, v in sorted(os.environ.items()) if _ENV.match(k) and k != "XLA_FLAGS"}
  return {
      "xla_flags": os.environ.get("XLA_FLAGS", "").split(),
      "env": env,
      "args": vars(args) if args is not None else None,
  }


def _amdsmi():
  """amd-smi's Python library, from ROCm or TheRock's wheels; None if absent."""
  try:
    import amdsmi  # pylint: disable=g-import-not-at-top
    return amdsmi
  except ImportError:
    pass
  candidates = ["/opt/rocm/share/amd_smi"]
  spec = importlib.util.find_spec("_rocm_sdk_core")
  if spec and spec.submodule_search_locations:
    candidates.insert(0, os.path.join(spec.submodule_search_locations[0], "share/amd_smi"))
  for path in candidates:
    if os.path.isdir(os.path.join(path, "amdsmi")):
      sys.path.insert(0, path)
      try:
        import amdsmi  # pylint: disable=g-import-not-at-top
        return amdsmi
      except ImportError:
        sys.path.remove(path)
  return None


def _gpus_amdsmi(amdsmi) -> list[dict]:
  amdsmi.amdsmi_init()
  try:
    out = []
    for h in amdsmi.amdsmi_get_processor_handles():
      gpu = {"pci_bus_id": amdsmi.amdsmi_get_gpu_device_bdf(h)}
      for key, fn in [
          ("uuid", lambda: amdsmi.amdsmi_get_gpu_device_uuid(h)),
          ("asic", lambda: amdsmi.amdsmi_get_gpu_asic_info(h)),
          ("vbios", lambda: amdsmi.amdsmi_get_gpu_vbios_info(h)),
          ("driver", lambda: amdsmi.amdsmi_get_gpu_driver_info(h)),
          ("power_cap_w", lambda: amdsmi.amdsmi_get_power_cap_info(h)["power_cap"] / 1e6),
      ]:
        try:
          gpu[key] = fn()
        except Exception:  # pylint: disable=broad-except
          pass
      asic = gpu.pop("asic", {})
      gpu.update({k: asic[k] for k in ("market_name", "num_compute_units", "target_graphics_version")
                  if k in asic})
      gpu["numa_node"] = _read(f"/sys/bus/pci/devices/{gpu['pci_bus_id']}/numa_node")
      out.append(gpu)
    return out
  finally:
    amdsmi.amdsmi_shut_down()


def _gpus_sysfs() -> list[dict]:
  out = []
  for dev in sorted(pathlib.Path("/sys/class/drm").glob("card*/device")):
    if not (dev / "hwmon").is_dir() or _read(dev / "vendor") != "0x1002":
      continue
    uevent = dict(line.split("=", 1) for line in (_read(dev / "uevent") or "").splitlines() if "=" in line)
    gpu = {"pci_bus_id": uevent.get("PCI_SLOT_NAME"), "vbios": _read(dev / "vbios_version"),
           "numa_node": _read(dev / "numa_node")}
    cap = next(iter(dev.glob("hwmon/hwmon*/power1_cap")), None)
    if cap and _read(cap):
      gpu["power_cap_w"] = int(_read(cap)) / 1e6
    out.append(gpu)
  return out


def _nics() -> list[dict]:
  out = []
  for dev in sorted(pathlib.Path("/sys/class/infiniband").glob("*")):
    nic = {"name": dev.name, "fw_ver": _read(dev / "fw_ver"), "hca_type": _read(dev / "hca_type"),
           "board_id": _read(dev / "board_id"), "ports": {}}
    for port in sorted((dev / "ports").glob("*")):
      nic["ports"][port.name] = {k: _read(port / k) for k in ("state", "rate", "link_layer")}
    nic["netdevs"] = sorted(p.name for p in (dev / "device/net").glob("*"))
    out.append(nic)
  return out


def hardware() -> dict:
  cpu = _read("/proc/cpuinfo") or ""
  model = next((line.split(":", 1)[1].strip() for line in cpu.splitlines()
                if line.startswith("model name")), platform.processor())
  amdsmi = _amdsmi()
  try:
    gpus = _gpus_amdsmi(amdsmi) if amdsmi else _gpus_sysfs()
  except Exception:  # pylint: disable=broad-except
    gpus = _gpus_sysfs()
  meminfo = _read("/proc/meminfo") or ""
  mem = re.search(r"MemTotal:\s+(\d+) kB", meminfo)
  return {
      "host": socket.gethostname(),
      "kernel": platform.release(),
      "os": next((line.split("=", 1)[1].strip('"') for line in (_read("/etc/os-release") or "").splitlines()
                  if line.startswith("PRETTY_NAME=")), None),
      "image": os.environ.get("JAXTITAN_IMAGE"),
      "cpu": {"model": model, "logical_cpus": os.cpu_count(),
              "numa_nodes": len(list(pathlib.Path("/sys/devices/system/node").glob("node[0-9]*")))},
      "memory_gib": int(mem.group(1)) / 2**20 if mem else None,
      "amdgpu_driver": _read("/sys/module/amdgpu/version"),
      "gpus": gpus,
      "nics": _nics(),
  }


def _hip_pci_bus_ids(count: int) -> list[str | None]:
  """PCI bus ID of each HIP device index, from the process's libamdhip64."""
  path = None
  for line in (_read("/proc/self/maps") or "").splitlines():
    if "libamdhip64.so" in line:
      path = line.split()[-1]
      break
  if not path:
    return [None] * count
  try:
    hip = ctypes.CDLL(path)
  except OSError:
    return [None] * count
  out = []
  for i in range(count):
    buf = ctypes.create_string_buffer(32)
    ok = hip.hipDeviceGetPCIBusId(buf, ctypes.c_int(32), ctypes.c_int(i)) == 0
    out.append(buf.value.decode().lower() if ok else None)
  return out


def rank_map(devices, placement=None) -> list[dict]:
  """Each local device: id, process, HIP index, kind, PCI bus ID, mesh and stages."""
  import jax  # pylint: disable=g-import-not-at-top

  local = [d for d in devices if d.process_index == jax.process_index()]
  on_gpu = local and local[0].platform in ("gpu", "rocm") and local[0].local_hardware_id is not None
  bus_ids = _hip_pci_bus_ids(max(d.local_hardware_id for d in local) + 1) if on_gpu else []
  mesh_of, stages_of = {}, {}
  if placement:
    for name, ids in placement["meshes"].items():
      for i in ids:
        mesh_of[i] = name
    for stage, name in enumerate(placement["stages"]):
      for i in placement["meshes"][name]:
        stages_of.setdefault(i, []).append(stage)
  out = []
  for d in local:
    entry = {
        "id": int(d.id), "process_index": int(d.process_index), "host": socket.gethostname(),
        "pid": os.getpid(), "kind": d.device_kind,
        "local_hardware_id": d.local_hardware_id,
    }
    if bus_ids and d.local_hardware_id is not None and d.local_hardware_id < len(bus_ids):
      entry["pci_bus_id"] = bus_ids[d.local_hardware_id]
    if d.id in mesh_of:
      entry["mesh"] = mesh_of[d.id]
      entry["stages"] = stages_of.get(d.id, [])
    out.append(entry)
  return out


def collect(args=None, devices=None, placement=None) -> dict:
  env = {"software": software(), "configuration": configuration(args), "hardware": hardware()}
  if devices is not None:
    env["rank_map"] = rank_map(devices, placement)
    env["placement"] = placement
  return env


def write(path, **kwargs) -> dict:
  env = collect(**kwargs)
  pathlib.Path(path).write_text(json.dumps(env, indent=1, default=str) + "\n")
  return env


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("out")
  args = parser.parse_args(argv)
  env = write(args.out)
  print(f"wrote {args.out}: {len(env['hardware']['gpus'])} GPUs, "
        f"{len(env['software']['packages'])} packages")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
