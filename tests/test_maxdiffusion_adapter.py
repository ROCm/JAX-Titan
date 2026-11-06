"""MaxDiffusion FLUX.1 stages match MaxDiffusion and single-mesh `jax.jit`."""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("maxdiffusion") is None, reason="needs MaxDiffusion"
)


@pytest.mark.parametrize("flags", [
    [],
    ["--blocks=3,1"],
    ["--num_stages=4", "--microbatches=8", "--schedule=GPIPE"],
    ["--flavor=flux-schnell"],
], ids=lambda f: " ".join(f) or "flux-dev")
def test_maxdiffusion_flux_parity(flags):
  result = subprocess.run(
      [sys.executable, str(REPO / "examples/maxdiffusion_flux_parity.py"), "--platform=cpu",
       *flags],
      capture_output=True, text=True, check=False,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  assert "match the transformer" in result.stdout
  assert "matches the single-mesh reference" in result.stdout
