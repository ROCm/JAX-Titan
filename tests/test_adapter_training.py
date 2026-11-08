"""MaxText and MaxDiffusion models train through the trainer and resume exactly."""

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]
FLAGS = [
    "--platform=cpu", "--microbatches=4", "--microbatch_size=4", "--steps=4",
    "--warmup_steps=1", "--checkpoint_every=2",
]
MAXTEXT_OVERRIDES = {
    "base_emb_dim": 64, "base_mlp_dim": 128, "base_num_query_heads": 4,
    "base_num_kv_heads": 2, "head_dim": 16, "base_num_decoder_layers": 4,
    "vocab_size": 256,
}
MODELS = {
    "maxtext": [
        "--model=maxtext:llama3.1-8b", "--data=synthetic", "--seq_len=32",
        "--model_overrides=" + json.dumps(MAXTEXT_OVERRIDES),
    ],
    "maxdiffusion": [
        "--model=maxdiffusion:flux-dev", "--resolution=64", "--text_tokens=8",
        "--model_overrides=" + json.dumps({
            "in_channels": 16, "num_layers": 2, "num_single_layers": 2,
            "attention_head_dim": 16, "num_attention_heads": 2,
            "joint_attention_dim": 32, "pooled_projection_dim": 16,
            "axes_dims_rope": [4, 6, 6],
        }),
    ],
}


def _run(provider, log, *flags):
  # os.environ, not the process environment: see conftest's ROCPROFILER note.
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", *FLAGS, *MODELS[provider],
       f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r for r in records.steps(records.read(log))}, result.stdout


def _train(provider, ckpt_dir, log):
  records, out = _run(provider, log, f"--checkpoint_dir={ckpt_dir}")
  return {step: r["loss"] for step, r in records.items()}, out


@pytest.mark.parametrize("provider", sorted(MODELS))
def test_resume_is_exact(provider, tmp_path):
  if importlib.util.find_spec(provider) is None:
    pytest.skip(f"needs {provider}")
  full, _ = _train(provider, tmp_path / "full", tmp_path / "full.jsonl")
  (tmp_path / "resumed").mkdir()
  shutil.copytree(tmp_path / "full" / "2", tmp_path / "resumed" / "2")
  resumed, out = _train(provider, tmp_path / "resumed", tmp_path / "resumed.jsonl")
  assert "restored step 2" in out
  assert sorted(resumed) == [2, 3]
  for step in (2, 3):
    assert resumed[step] == full[step], (step, resumed[step], full[step])


def test_maxtext_starts_from_a_parameter_checkpoint(tmp_path):
  if importlib.util.find_spec("maxtext") is None:
    pytest.skip("needs maxtext")
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top
  from maxtext.common import checkpointing  # pylint: disable=g-import-not-at-top

  # The trainer's own initialization, saved as MaxText's parameter-only checkpoint.
  cfg = mt.config("llama3.1-8b", max_target_length=32, remat_policy="full", **MAXTEXT_OVERRIDES)
  model = mt.Model(cfg, mt.mesh(cfg, jax.devices()[:4]), num_stages=2)
  full = jax.jit(lambda: model._init_full(0)[0])()  # pylint: disable=protected-access
  path = str(tmp_path / "params")
  checkpointing.save_params_to_path(path, {"params": full})
  for a, b in zip(jax.tree.leaves(model.load(path)), jax.tree.leaves(model.split(full))):
    np.testing.assert_array_equal(a, b)

  base, _ = _run("maxtext", tmp_path / "base.jsonl")
  loaded, _ = _run("maxtext", tmp_path / "loaded.jsonl", f"--init_from={path}")
  assert loaded[0]["loss"] == base[0]["loss"]


@pytest.mark.parametrize("provider", sorted(MODELS))
def test_pipeline_matches_no_pipeline_reference(provider, tmp_path):
  if importlib.util.find_spec(provider) is None:
    pytest.skip(f"needs {provider}")
  pipeline, _ = _run(provider, tmp_path / "mpmd.jsonl")
  reference, out = _run(provider, tmp_path / "dp.jsonl", "--parallel=dp")
  assert "no pipeline" in out
  # Same weights and batch: step 0's loss agrees to rounding; bf16 gradient
  # reductions in a different order move later steps apart.
  assert pipeline[0]["loss"] == pytest.approx(reference[0]["loss"], rel=1e-6)
  for step in range(4):
    assert pipeline[step]["loss"] == pytest.approx(reference[step]["loss"], rel=1e-3), step
    assert pipeline[step]["grad_norm"] == pytest.approx(reference[step]["grad_norm"], rel=5e-3), step
