"""MaxText and MaxDiffusion models train through the trainer and resume exactly."""

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
FLAGS = [
    "--platform=cpu", "--microbatches=4", "--microbatch_size=4", "--steps=4",
    "--warmup_steps=1", "--checkpoint_every=2",
]
MODELS = {
    "maxtext": [
        "--model=maxtext:llama3.1-8b", "--data=synthetic", "--seq_len=32",
        "--model_overrides=" + json.dumps({
            "base_emb_dim": 64, "base_mlp_dim": 128, "base_num_query_heads": 4,
            "base_num_kv_heads": 2, "head_dim": 16, "base_num_decoder_layers": 4,
            "vocab_size": 256,
        }),
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


def _train(provider, ckpt_dir, log):
  result = subprocess.run(
      [sys.executable, "-m", "jaxtitan.trainer.train", *FLAGS, *MODELS[provider],
       f"--checkpoint_dir={ckpt_dir}", f"--log_file={log}"],
      capture_output=True, text=True, check=False, cwd=REPO,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r["loss"] for r in map(json.loads, open(log))}, result.stdout


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
