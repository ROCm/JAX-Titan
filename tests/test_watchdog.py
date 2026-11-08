"""The step watchdog ends a stuck step and leaves the others alone."""

import subprocess
import sys
import textwrap

from jax_titan.observability import watchdog


def _run(code):
  return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True,
                        text=True, timeout=120, check=False)


def test_exits_when_a_step_hangs():
  result = _run("""
      import time
      from jax_titan.observability.watchdog import StepWatchdog
      watchdog = StepWatchdog(0.5, poll_s=0.05)
      watchdog.start(0)
      time.sleep(30)
  """)
  assert result.returncode == watchdog.EXIT_CODE, result.stderr
  assert "step watchdog: step 0 has run for" in result.stderr


def test_later_steps_get_a_multiple_of_the_median():
  result = _run("""
      import time
      from jax_titan.observability.watchdog import StepWatchdog
      watchdog = StepWatchdog(0.5, poll_s=0.05)
      for step in range(5):
        watchdog.start(step)
        time.sleep(0.1)
        watchdog.stop()
      watchdog.start(5)
      time.sleep(0.8)  # past the 0.5 s floor, within 10x the 0.1 s median
      watchdog.stop()
      print("done")
  """)
  assert result.returncode == 0, result.stderr
  assert "done" in result.stdout
