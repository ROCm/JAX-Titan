"""The MPMD pipeline step matches single-mesh `jax.jit` in every layout."""

import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _run(script, *flags):
  result = subprocess.run(
      [sys.executable, str(REPO / script), "--platform=cpu", *flags],
      capture_output=True, text=True, check=False,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return result.stdout


def test_two_mesh_spike(tmp_path):
  out = _run("examples/mpmd_two_mesh_spike.py", f"--dump_dir={tmp_path}")
  assert "params match the single-mesh reference" in out


@pytest.mark.parametrize("flags", [
    ["--reduce_grads=step"],
    ["--reduce_grads=microbatch"],
    ["--reduce_grads=step", "--stage_fsdp"],
    ["--reduce_grads=microbatch", "--stage_fsdp"],
    ["--reduce_grads=step", "--tensor_parallel=2"],
    ["--reduce_grads=microbatch", "--stage_fsdp", "--tensor_parallel=2"],
    ["--schedule=GPIPE"],
    ["--num_stages=4", "--microbatches=8"],
    ['--mpmd_options={"mpmd_merge_inferred_after_scheduling": true}'],
    ["--schedule=ZERO_BUBBLE_H1"],
    ["--schedule=CIRCULAR", "--num_stages=4", "--stages_per_mesh=2"],
    ["--stage_layers=3,1"],
    ["--reduce_grads=step", "--per_device=shard_map"],
    ["--reduce_grads=step", "--per_device=shard_map", "--stage_fsdp", "--tensor_parallel=2"],
], ids=lambda f: " ".join(f))
def test_llama_parity(flags):
  out = _run("examples/llama_mpmd_parity.py", *flags)
  assert "matches the single-mesh reference" in out
