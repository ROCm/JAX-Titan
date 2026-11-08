"""Ends a training process whose step stops making progress.

A collective that never completes, such as RCCL communicator setup while
another job holds the GPUs, blocks the host for good: XLA's execution timeouts
are off by default. The watchdog runs on its own thread, so it fires while the
main thread waits on the device.
"""

import faulthandler
import os
import statistics
import sys
import threading
import time

EXIT_CODE = 124
FACTOR = 10


class StepWatchdog:
  """Exits with `EXIT_CODE` when a step runs past its limit.

  The first step's limit is `timeout_s` seconds; it includes communicator
  setup. A later step's limit is `FACTOR` times the median step so far, but at
  least `timeout_s`. `start` and `stop` bracket the step's execution only.
  """

  def __init__(self, timeout_s: float, poll_s: float = 1.0):
    self._timeout_s, self._poll_s = timeout_s, poll_s
    self._times: list[float] = []
    self._step = self._started = self._deadline = None
    self._lock = threading.Lock()
    threading.Thread(target=self._watch, name="step-watchdog", daemon=True).start()

  def start(self, step: int) -> None:
    limit = self._timeout_s
    if self._times:
      limit = max(limit, FACTOR * statistics.median(self._times))
    now = time.monotonic()
    with self._lock:
      self._step, self._started, self._deadline = step, now, now + limit

  def stop(self) -> None:
    with self._lock:
      if self._started is not None:
        self._times.append(time.monotonic() - self._started)
      self._step = self._started = self._deadline = None

  def _watch(self) -> None:
    while True:
      time.sleep(self._poll_s)
      with self._lock:
        step, started, deadline = self._step, self._started, self._deadline
      if deadline is not None and time.monotonic() > deadline:
        print(f"step watchdog: step {step} has run for {time.monotonic() - started:.0f} s, "
              f"past its limit of {deadline - started:.0f} s; exiting", file=sys.stderr,
              flush=True)
        faulthandler.dump_traceback(all_threads=True)
        os._exit(EXIT_CODE)
