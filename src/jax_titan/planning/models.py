"""Decoder-only transformer shapes for training budgets, dense or mixture-of-experts.

A `Model` comes from the model config index (`data/models/model_config_index.json`: public
Hugging Face config.json fields, by name or repository), a downloaded Hugging Face model
(`hf:<config.json or its directory>`), a MaxText model (`maxtext:<name>`, from the index
when it has the name), a JAX-Titan Llama flavor, or explicit dimensions. FLOPs follow
`jax_titan.observability.flops`: two FLOPs per multiply-add, a training step costs three
forward passes (a matmul's backward computes the activation and the weight gradient), and
causal attention counts half of its scores. Attention is grouped-query; multi-head latent
attention (DeepSeek's MLA) is not modeled.
"""

import dataclasses
import json
import os
import pathlib

from jax_titan.planning import model_index

# How much of the attention scores (QK^T and AV) a convention counts: JAX-Titan's causal
# half, PaLM's and Megatron's full square, or none (the 6N of published compute figures).
ATTENTION = {"causal": 1, "full": 2, "none": 0}


@dataclasses.dataclass(frozen=True)
class Model:
  name: str
  vocab: int
  emb: int
  layers: int
  q_heads: int
  kv_heads: int
  head_dim: int
  mlp: int  # FFN width of dense layers
  experts: int = 0  # routed experts per MoE layer; 0 for a dense model
  experts_per_token: int = 0
  expert_mlp: int = 0  # FFN width of one expert
  shared_experts: int = 0  # experts every token uses, each `expert_mlp` wide
  moe_layers: int = 0  # how many of `layers` are MoE layers
  tied_embeddings: bool = False
  source: str = ""  # where the shape came from: repository@commit, a path, or a config

  @property
  def dense_layers(self) -> int:
    return self.layers - self.moe_layers

  def attention_params(self) -> int:
    return self.emb * self.head_dim * (2 * self.q_heads + 2 * self.kv_heads)

  def expert_params(self) -> int:
    return 3 * self.emb * self.expert_mlp

  def mlp_params(self, moe: bool, active: bool = False) -> int:
    if not moe:
      return 3 * self.emb * self.mlp
    routed = self.experts_per_token if active else self.experts
    return (routed + self.shared_experts) * self.expert_params() + self.emb * self.experts

  def embedding_params(self) -> int:
    return self.vocab * self.emb * (1 if self.tied_embeddings else 2)

  def params(self, active: bool = False) -> int:
    """All parameters, or those one token uses (`active`) in an MoE model."""
    per_layer_norms = 2 * self.emb
    dense = self.dense_layers * (self.attention_params() + self.mlp_params(False) + per_layer_norms)
    moe = self.moe_layers * (self.attention_params() + self.mlp_params(True, active)
                             + per_layer_norms)
    return dense + moe + self.embedding_params() + self.emb

  def expert_param_count(self) -> int:
    """Parameters that expert parallelism shards: the routed experts."""
    return self.moe_layers * self.experts * self.expert_params()

  def forward_flops_per_token(self, seq_len: int) -> dict[str, float]:
    """Forward FLOPs per token over the whole model, per op."""
    e, h, k, hd = self.emb, self.q_heads, self.kv_heads, self.head_dim
    out = {
        "attention_projections": self.layers * 2 * e * hd * (2 * h + 2 * k),
        # QK^T and AV over the causal half of seq_len keys.
        "attention_scores": self.layers * 2 * h * hd * seq_len,
        "mlp": self.dense_layers * 2 * 3 * e * self.mlp,
        "lm_head": 2 * e * self.vocab,
    }
    if self.moe_layers:
      out["router"] = self.moe_layers * 2 * e * self.experts
      out["experts"] = self.moe_layers * 2 * (
          self.experts_per_token + self.shared_experts) * self.expert_params()
    return out

  def _forward(self, seq_len: int, attention: str) -> dict[str, float]:
    f = self.forward_flops_per_token(seq_len)
    f["attention_scores"] *= ATTENTION[attention]
    return f

  def training_flops_per_token(self, seq_len: int, attention: str = "causal") -> float:
    """Model FLOPs per trained token, the numerator of MFU, with the attention scores
    counted per `ATTENTION`."""
    return 3 * sum(self._forward(seq_len, attention).values())

  def recompute_flops_per_token(self, seq_len: int, remat: bool = True,
                                attention: str = "causal") -> float:
    """FLOPs that rematerializing every decoder layer adds: their forward pass again."""
    if not remat:
      return 0.0
    f = self._forward(seq_len, attention)
    return sum(v for k, v in f.items() if k != "lm_head")


def from_llama(cfg, name: str = "llama") -> Model:
  """From a `jax_titan.models.llama.LlamaConfig`."""
  return Model(name, cfg.vocab_size, cfg.emb_dim, cfg.num_layers, cfg.num_query_heads,
               cfg.num_kv_heads, cfg.head_dim, cfg.mlp_dim, source="jax_titan.models.llama")


def cpu_only() -> None:
  """Keeps JAX on the CPU; call before anything imports jax. Planning needs no GPU,
  but reading a MaxText config counts devices, which otherwise starts JAX's GPU
  backend: a context on every GPU and, by default, 75% of its memory."""
  os.environ.setdefault("JAX_PLATFORMS", "cpu")


def from_maxtext(model_name: str) -> Model:
  """From a MaxText model config (needs MaxText installed)."""
  from jax_titan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top

  cfg = mt.config(model_name)
  if str(getattr(cfg, "attention_type", "")).lower().endswith("mla"):
    raise NotImplementedError(f"{model_name} uses multi-head latent attention")
  experts = int(getattr(cfg, "num_experts", 1) or 1)
  moe = experts > 1
  dense_first = int(getattr(cfg, "first_num_dense_layers", 0) or 0)
  step = int(getattr(cfg, "interleave_moe_layer_step", 1) or 1)
  layers = cfg.num_decoder_layers
  moe_layers = ((layers - dense_first) // step) if moe else 0
  return Model(
      f"maxtext:{model_name}", cfg.vocab_size, cfg.emb_dim, layers, cfg.num_query_heads,
      cfg.num_kv_heads, cfg.head_dim, cfg.mlp_dim,
      experts=experts if moe else 0,
      experts_per_token=int(getattr(cfg, "num_experts_per_tok", 0) or 0) if moe else 0,
      expert_mlp=int(getattr(cfg, "moe_mlp_dim", 0) or cfg.mlp_dim) if moe else 0,
      shared_experts=int(getattr(cfg, "shared_experts", 0) or 0) if moe else 0,
      moe_layers=moe_layers,
      tied_embeddings=bool(getattr(cfg, "logits_via_embedding", False)),
      source=f"MaxText configs/models/{model_name}.yml",
  )


def from_hf_dict(c: dict, name: str, source: str = "") -> Model:
  """From the fields of a Hugging Face `config.json` of a Llama-, Mistral-, Mixtral-,
  Qwen- or DeepSeek-style (without MLA) decoder."""
  c = c.get("text_config", c)
  if c.get("kv_lora_rank"):
    raise NotImplementedError(f"{name} uses multi-head latent attention (kv_lora_rank)")
  heads = c["num_attention_heads"]
  emb = c["hidden_size"]
  layers = c["num_hidden_layers"]
  experts = c.get("n_routed_experts") or c.get("num_local_experts") or c.get("num_experts") or 0
  moe_layers = 0
  if experts:
    dense_first = c.get("first_k_dense_replace", 0) or 0
    step = c.get("decoder_sparse_step", 1) or 1
    dense_only = set(c.get("mlp_only_layers") or [])
    moe_layers = len([i for i in range(dense_first, layers)
                      if (i + 1) % step == 0 and i not in dense_only])
  shared = c.get("n_shared_experts") or 0
  expert_mlp = c.get("moe_intermediate_size") or c.get("intermediate_size")
  if c.get("shared_expert_intermediate_size") and not shared:
    # One shared expert of its own width (Qwen-MoE): count it in units of expert_mlp.
    shared = c["shared_expert_intermediate_size"] / expert_mlp
  return Model(
      name, c["vocab_size"], emb, layers, heads,
      c.get("num_key_value_heads") or heads, c.get("head_dim") or emb // heads,
      c.get("intermediate_size") or 4 * emb,
      experts=experts, experts_per_token=c.get("num_experts_per_tok", 0) if experts else 0,
      expert_mlp=expert_mlp if experts else 0, shared_experts=shared,
      moe_layers=moe_layers, tied_embeddings=bool(c.get("tie_word_embeddings", False)),
      source=source,
  )


def from_hf_config(path) -> Model:
  """From a downloaded model's `config.json`, or the directory that holds it."""
  path = pathlib.Path(path)
  if path.is_dir():
    path = path / "config.json"
  return from_hf_dict(json.loads(path.read_text()), path.parent.name or str(path),
                      source=str(path))


def lookup(name: str) -> str | None:
  """The index entry for `name`, in any case: its key, or a Hugging Face repository
  (with or without the owner) that the entry was read from or mirrors."""
  name = name.lower()
  index = model_index.load()
  if name in index:
    return name
  for key, entry in index.items():
    for repo in (entry["hf_config_source"], entry.get("mirror_of", "")):
      if repo and name in (repo.lower(), repo.lower().rsplit("/", 1)[-1]):
        return key
  return None


def from_index(key: str) -> Model:
  entry = model_index.load()[key]
  return from_hf_dict(entry["config"], key, source=(f"{entry['hf_config_source']}@"
                                                    f"{entry['hf_config_revision'][:12]}"))


def from_name(name: str) -> Model:
  """A downloaded model (a path to its config.json or the directory holding it, with
  or without `hf:`), a model in the index (by name or Hugging Face repository),
  `maxtext:<name>` (from the index when it has the name, else from MaxText's config),
  or a JAX-Titan Llama flavor. A path that exists comes first."""
  if name.startswith("hf:"):
    return from_hf_config(name.split(":", 1)[1])
  local = pathlib.Path(name).expanduser()
  if local.is_file() or (local / "config.json").is_file():
    return from_hf_config(local)
  if name.startswith("maxtext:"):
    key = lookup(name.split(":", 1)[1])
    return from_index(key) if key else from_maxtext(name.split(":", 1)[1])
  key = lookup(name)
  if key:
    return from_index(key)
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top

  if name not in llama.FLAVORS:
    raise ValueError(
        f"unknown model {name!r}: a path to a downloaded config.json or its directory, one "
        f"of {', '.join(model_index.load())} (or its Hugging Face repository), "
        f"maxtext:<name>, or a Llama flavor ({', '.join(llama.FLAVORS)})")
  return from_llama(llama.FLAVORS[name], name)
