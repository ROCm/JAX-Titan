"""Default locations, overridable with environment variables.

`JAX_TITAN_ROOT` holds source worktrees, the Bazel output root, the venv, data,
caches and logs. It defaults to a `.jax_titan` directory next to this repository.
"""

import os
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]


def work_dir() -> pathlib.Path:
  return pathlib.Path(os.environ.get("JAX_TITAN_ROOT", _REPO.parent / ".jax_titan"))


def llama3_tokenizer() -> pathlib.Path:
  default = (
      _REPO.parent / "maxtext/src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken"
  )
  return pathlib.Path(os.environ.get("JAX_TITAN_LLAMA3_TOKENIZER", default))
