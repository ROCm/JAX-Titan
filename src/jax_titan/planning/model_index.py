"""The model config index: public Hugging Face `config.json` fields per model.

`jax_titan.planning.models` resolves `--model` names against
`data/models/model_config_index.json`, so planning needs neither MaxText nor a
download. Each entry records the repository and commit its fields were read
from (https://huggingface.co/<hf_config_source>/blob/<hf_config_revision>/config.json);
`mirror_of` names the gated original that an ungated copy reproduces.

  python -m jax_titan.planning.model_index           # the models in the index
  python -m jax_titan.planning.model_index --add qwen3-14b Qwen/Qwen3-14B
  python -m jax_titan.planning.model_index --add llama3.1-8b unsloth/Meta-Llama-3.1-8B \\
      --mirror_of meta-llama/Llama-3.1-8B

`--add` reads REPO's config.json from the Hugging Face Hub and pins the
repository's latest commit, unless `--revision` names one. With HF_TOKEN set it
reads gated repositories directly. A model you downloaded needs no entry: plan
it with `--model=<config.json or its directory>`.
"""

import argparse
import json
import os
import pathlib
import urllib.request

PATH = pathlib.Path(__file__).with_name("data") / "models" / "model_config_index.json"

# The config.json fields that planning reads, and the ones that name the architecture.
FIELDS = (
    "model_type", "architectures", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
    "vocab_size", "tie_word_embeddings", "max_position_embeddings",
    "num_local_experts", "num_experts", "n_routed_experts", "num_experts_per_tok",
    "moe_intermediate_size", "shared_expert_intermediate_size", "n_shared_experts",
    "first_k_dense_replace", "decoder_sparse_step", "mlp_only_layers", "kv_lora_rank",
)


def load() -> dict:
  return json.loads(PATH.read_text())["models"]


def _get(url: str) -> bytes:
  request = urllib.request.Request(url)
  if os.environ.get("HF_TOKEN"):
    request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
  with urllib.request.urlopen(request, timeout=60) as response:
    return response.read()


def fetch(repo: str, revision: str = "") -> tuple[str, dict]:
  """The commit, and the FIELDS of `repo`'s config.json at it."""
  revision = revision or json.loads(_get(f"https://huggingface.co/api/models/{repo}"))["sha"]
  config = json.loads(_get(f"https://huggingface.co/{repo}/resolve/{revision}/config.json"))
  config = config.get("text_config", config)
  return revision, {k: config[k] for k in FIELDS if k in config}


def add(key: str, repo: str, mirror_of: str = "", revision: str = "") -> dict:
  revision, config = fetch(repo, revision)
  entry = {"hf_config_source": repo, "hf_config_revision": revision}
  if mirror_of:
    entry["mirror_of"] = mirror_of
  entry["config"] = config
  index = json.loads(PATH.read_text())
  index["models"] = dict(sorted({**index["models"], key: entry}.items()))
  PATH.write_text(json.dumps(index, indent=1) + "\n")
  return entry


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--add", nargs=2, metavar=("NAME", "REPO"),
                      help="Add or replace NAME with the config.json fields of REPO, a Hugging "
                      "Face repository (owner/name), read from the Hub.")
  parser.add_argument("--mirror_of", default="",
                      help="With --add: the gated repository REPO copies.")
  parser.add_argument("--revision", default="", help="With --add: the commit to pin.")
  args = parser.parse_args(argv)
  if args.add and pathlib.Path(args.add[1]).exists():
    parser.error(f"{args.add[1]} is a local path; REPO is a Hugging Face repository such as "
                 "Qwen/Qwen3-14B-Base. A downloaded model needs no index entry: plan it with "
                 "python -m jax_titan.planning.budget --model=<config.json or its directory>")
  if args.add:
    entry = add(*args.add, mirror_of=args.mirror_of, revision=args.revision)
    print(f"{args.add[0]}: {entry['hf_config_source']}@{entry['hf_config_revision']}")
    return 0
  for name, e in load().items():
    mirror = f" (mirror of {e['mirror_of']})" if "mirror_of" in e else ""
    print(f"{name:18s} {e['hf_config_source']}@{e['hf_config_revision'][:12]}{mirror}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
