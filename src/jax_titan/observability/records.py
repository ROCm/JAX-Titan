"""The trainer's JSONL log: one record per step, and timeline events.

Step records hold `step`, `loss`, `grad_norm`, `step_time_s`, `tokens_per_s`,
`tokens_per_s_per_gpu`, `mfu` and `time`, the wall-clock time the step ended.
Event records hold `event` and `time`:

  start    `process_start`, `main_start` and `jax_ready` wall-clock times (and
           `launch_time` when the launcher exports JAX_TITAN_LAUNCH_TIME), host,
           pid, process index and count, devices, device kind and the args
  state    `seconds` to initialize or restore the training state, and
           `restored_step` (0 when initialized)
  compile  `lower_s` and `compile_s`
  data     `seconds` to open the data
  memory   `peak_hbm_gib` per device after the first step
  eval     `step` (updates done), validation `loss` and `seconds`
  save     `step` and `blocking_s`, how long the step loop waited for the save
  end      `status` ("completed", or the exception) and `steps_done`

A log appended across restarts holds one attempt per `start` event.
"""

import json
import os
import time


def process_start_time() -> float | None:
  """Wall-clock time this process started, from /proc; None elsewhere."""
  try:
    with open("/proc/self/stat") as f:
      fields = f.read().rsplit(")", 1)[1].split()
    with open("/proc/stat") as f:
      boot = next(int(line.split()[1]) for line in f if line.startswith("btime"))
    return boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")
  except (OSError, ValueError, IndexError, StopIteration):
    return None


class Log:
  """Appends records to a JSONL file; does nothing without a path."""

  def __init__(self, path: str | None):
    self._file = open(path, "a") if path else None

  def write(self, record: dict):
    if self._file:
      self._file.write(json.dumps(record) + "\n")
      self._file.flush()

  def event(self, name: str, **fields):
    self.write({"event": name, "time": time.time(), **fields})

  def close(self):
    if self._file:
      self._file.close()


def read(path) -> list[dict]:
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]


def is_step(record: dict) -> bool:
  return "event" not in record


def steps(records) -> list[dict]:
  return [r for r in records if is_step(r)]


def events(records, name: str) -> list[dict]:
  return [r for r in records if r.get("event") == name]


def attempts(records) -> list[list[dict]]:
  """Records split at each `start` event; records before the first start
  (logs from before timeline events) form one attempt of their own."""
  out = []
  for r in records:
    if r.get("event") == "start" or not out:
      out.append([])
    out[-1].append(r)
  return out
