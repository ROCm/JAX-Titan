"""GPU power, energy, clocks, temperature and throttling over a run.

  python -m jax_titan.observability.power OUT.jsonl [--interval 1] [--duration S]

Each line is one sample of one GPU: `time`, `gpu` (PCI bus ID), `energy_j`
(the GPU's accumulated energy counter), `power_w`, `gfx_mhz`, `mem_mhz` and
`temp_c` (hotspot). Every `violation_every` seconds a sample also holds
`throttle`: the share of the time since the previous reading that the GPU was
held back by its power limit (`ppt`), socket, VR or HBM temperature, or
PROCHOT. The counters come from amd-smi's library; without it, sysfs hwmon
gives power, clocks and temperature, and energy is power integrated over
time. The measurement boundary is the GPU. Sampling at 1 Hz cannot see
millisecond-scale power excursions.

The trainer samples into `<run_dir>/power.jsonl` with `--power_sample_s`.
"""

import argparse
import json
import pathlib
import threading
import time

from jax_titan.observability import env as env_lib

_THROTTLE = {
    "ppt": "acc_ppt_pwr", "socket_thermal": "acc_socket_thrm", "vr_thermal": "acc_vr_thrm",
    "hbm_thermal": "acc_hbm_thrm", "prochot": "acc_prochot_thrm",
}


def _number(value):
  """amd-smi reports unavailable readings as "N/A"."""
  return value if isinstance(value, (int, float)) else None


class _AmdSmi:
  name = "amd-smi energy counter"

  def __init__(self, amdsmi):
    self.a = amdsmi
    amdsmi.amdsmi_init()
    self.handles = amdsmi.amdsmi_get_processor_handles()
    self.ids = [amdsmi.amdsmi_get_gpu_device_bdf(h) for h in self.handles]

  def sample(self, h) -> dict:
    a = self.a
    energy = a.amdsmi_get_energy_count(h)
    power = a.amdsmi_get_power_info(h)
    return {
        "energy_j": energy["energy_accumulator"] * energy["counter_resolution"] / 1e6,
        "power_w": _number(power.get("socket_power")),
        "gfx_mhz": _number(a.amdsmi_get_clock_info(h, a.AmdSmiClkType.GFX)["clk"]),
        "mem_mhz": _number(a.amdsmi_get_clock_info(h, a.AmdSmiClkType.MEM)["clk"]),
        "temp_c": _number(a.amdsmi_get_temp_metric(
            h, a.AmdSmiTemperatureType.HOTSPOT, a.AmdSmiTemperatureMetric.CURRENT)),
    }

  def violation(self, h) -> dict:
    return self.a.amdsmi_get_violation_status(h)

  def close(self):
    self.a.amdsmi_shut_down()


class _Sysfs:
  name = "hwmon power, integrated"

  def __init__(self):
    self.devices = []
    for dev in sorted(pathlib.Path("/sys/class/drm").glob("card*/device")):
      hwmon = next(iter(dev.glob("hwmon/hwmon*")), None)
      if env_lib._read(dev / "vendor") != "0x1002" or not hwmon:  # pylint: disable=protected-access
        continue
      if (hwmon / "power1_average").exists() or (hwmon / "power1_input").exists():
        self.devices.append((dev, hwmon))
    self.handles = list(range(len(self.devices)))
    self.ids = [env_lib._read(d / "uevent").split("PCI_SLOT_NAME=")[1].split()[0]  # pylint: disable=protected-access
                for d, _ in self.devices]
    self._energy = [0.0] * len(self.devices)
    self._last = [None] * len(self.devices)

  def sample(self, h) -> dict:
    _, hwmon = self.devices[h]

    def value(name, scale):
      text = env_lib._read(hwmon / name)  # pylint: disable=protected-access
      return int(text) * scale if text else None

    power = value("power1_average", 1e-6) or value("power1_input", 1e-6)
    now = time.time()
    if self._last[h] is not None and power is not None:
      t, p = self._last[h]
      self._energy[h] += (now - t) * (p + power) / 2
    self._last[h] = (now, power)
    return {
        "energy_j": self._energy[h], "power_w": power, "gfx_mhz": value("freq1_input", 1e-6),
        "mem_mhz": value("freq2_input", 1e-6), "temp_c": value("temp2_input", 1e-3),
    }

  def violation(self, h):
    return None

  def close(self):
    pass


def _source():
  amdsmi = env_lib._amdsmi()  # pylint: disable=protected-access
  if amdsmi:
    try:
      return _AmdSmi(amdsmi)
    except Exception:  # pylint: disable=broad-except
      pass
  return _Sysfs()


class Sampler:
  """Samples every GPU on this host into a JSONL file from a daemon thread."""

  def __init__(self, path, interval: float = 1.0, violation_every: float = 10.0):
    self._path, self._interval, self._violation_every = path, interval, violation_every
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True, name="power-sampler")

  def start(self):
    self._thread.start()
    return self

  def stop(self):
    self._stop.set()
    self._thread.join(timeout=10 + self._interval)

  def _run(self):
    source = _source()
    last_violation, previous = 0.0, {}
    with open(self._path, "a") as f:
      while not self._stop.is_set():
        now = time.time()
        read_violation = now - last_violation >= self._violation_every
        for h, gpu in zip(source.handles, source.ids):
          try:
            record = {"time": time.time(), "gpu": gpu, "source": source.name, **source.sample(h)}
            if read_violation:
              v = source.violation(h)
              if v and gpu in previous:
                counter = v["acc_counter"] - previous[gpu]["acc_counter"]
                if counter > 0:
                  record["throttle"] = {
                      k: (v[f] - previous[gpu][f]) / counter for k, f in _THROTTLE.items()
                      if isinstance(v.get(f), (int, float))
                  }
              if v:
                previous[gpu] = v
          except Exception as e:  # pylint: disable=broad-except
            record = {"time": time.time(), "gpu": gpu, "error": f"{type(e).__name__}: {e}"}
          f.write(json.dumps(record) + "\n")
        f.flush()
        if read_violation:
          last_violation = now
        self._stop.wait(max(0.0, self._interval - (time.time() - now)))
    source.close()


def read(path) -> list[dict]:
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]


def _mean(values):
  values = [v for v in values if _number(v) is not None]
  return sum(values) / len(values) if values else None


def energy_between(samples, t0: float, t1: float) -> dict:
  """Per GPU: joules, mean watts, mean GFX clock, and throttle shares in [t0, t1].

  Energy is the counter difference between the samples nearest each end.
  """
  by_gpu = {}
  for s in samples:
    if "energy_j" in s and s["energy_j"] is not None:
      by_gpu.setdefault(s["gpu"], []).append(s)
  out = {}
  for gpu, ss in by_gpu.items():
    inside = [s for s in ss if t0 <= s["time"] <= t1]
    if len(inside) < 2:
      continue
    first, last = inside[0], inside[-1]
    seconds = last["time"] - first["time"]
    throttles = [s["throttle"] for s in inside if "throttle" in s]
    out[gpu] = {
        "energy_j": last["energy_j"] - first["energy_j"],
        "seconds": seconds,
        "mean_power_w": (last["energy_j"] - first["energy_j"]) / seconds if seconds else None,
        "mean_gfx_mhz": _mean(s.get("gfx_mhz") for s in inside),
        "max_temp_c": max((s["temp_c"] for s in inside if _number(s.get("temp_c")) is not None),
                          default=None),
        "throttle": {k: sum(t.get(k, 0) for t in throttles) / len(throttles)
                     for k in _THROTTLE} if throttles else None,
    }
  return out


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("out")
  parser.add_argument("--interval", type=float, default=1.0)
  parser.add_argument("--violation_every", type=float, default=10.0)
  parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 runs until killed.")
  args = parser.parse_args(argv)
  sampler = Sampler(args.out, args.interval, args.violation_every).start()
  try:
    if args.duration:
      time.sleep(args.duration)
    else:
      while True:
        time.sleep(3600)
  except KeyboardInterrupt:
    pass
  sampler.stop()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
