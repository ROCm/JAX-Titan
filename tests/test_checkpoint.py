"""Resuming from an Orbax checkpoint continues training exactly."""

import pathlib
import shutil
import subprocess
import sys

import numpy as np

from jaxtitan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]
FLAGS = [
    "--platform=cpu", "--model=llama3-test", "--data=synthetic", "--seq_len=64",
    "--microbatches=4", "--microbatch_size=4", "--steps=6", "--warmup_steps=2",
    "--checkpoint_every=3",
]


def _train(ckpt_dir, log, *flags):
  result = subprocess.run(
      [sys.executable, "-m", "jaxtitan.trainer.train", *FLAGS,
       f"--checkpoint_dir={ckpt_dir}", f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r["loss"] for r in records.steps(records.read(log))}, result.stdout


def _resume_from_step_3(tmp_path, name, *flags):
  full, _ = _train(tmp_path / "full", tmp_path / "full.jsonl")
  (tmp_path / name).mkdir()
  shutil.copytree(tmp_path / "full" / "3", tmp_path / name / "3")
  resumed, out = _train(tmp_path / name, tmp_path / f"{name}.jsonl", *flags)
  assert "restored step 3" in out
  assert sorted(resumed) == [3, 4, 5]
  return full, resumed


def test_resume_is_exact(tmp_path):
  full, resumed = _resume_from_step_3(tmp_path, "resumed")
  for step in (3, 4, 5):
    assert resumed[step] == full[step], (step, resumed[step], full[step])


def test_restore_into_fsdp_stages(tmp_path):
  full, resumed = _resume_from_step_3(tmp_path, "fsdp", "--stage_fsdp")
  for step in (3, 4, 5):
    np.testing.assert_allclose(resumed[step], full[step], rtol=1e-5)
