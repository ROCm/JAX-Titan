"""MaxText Llama stages match MaxText, `jaxtitan.models.llama` and `jax.jit`."""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("maxtext") is None, reason="needs the maxtext extra"
)


@pytest.mark.parametrize("flags", [
    [],
    ["--stage_layers=3,1"],
    ["--num_stages=4", "--microbatches=8", "--schedule=GPIPE"],
    ["--model=llama3-8b"],
], ids=lambda f: " ".join(f) or "llama3.1-8b")
def test_maxtext_llama_parity(flags):
  result = subprocess.run(
      [sys.executable, str(REPO / "examples/maxtext_llama_parity.py"), "--platform=cpu", *flags],
      capture_output=True, text=True, check=False,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  assert "match MaxText's Transformer" in result.stdout
  assert "jaxtitan.models.llama matches the MaxText stages" in result.stdout
  assert "matches the single-mesh reference" in result.stdout
