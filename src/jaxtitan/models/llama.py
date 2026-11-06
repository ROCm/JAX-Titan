"""Llama 3 decoder with parameters grouped by pipeline stage.

Architecture and flavors follow MaxText's `decoder_block: llama2` and its
`configs/models/llama3*.yml`: pre-norm RMSNorm, RoPE, grouped-query attention
and a SwiGLU MLP.
"""

from collections.abc import Sequence
import dataclasses

import jax
import jax.numpy as jnp

from jaxtitan import ops


@dataclasses.dataclass(frozen=True)
class Llama31RopeScaling:
  """Llama 3.1's RoPE frequency scaling for contexts past `original_context`."""

  factor: float = 8.0
  low_freq_factor: float = 1.0
  high_freq_factor: float = 4.0
  original_context: int = 8192


@dataclasses.dataclass(frozen=True)
class LlamaConfig:
  vocab_size: int
  emb_dim: int
  num_layers: int
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  mlp_dim: int
  norm_eps: float = 1e-5
  rope_theta: float = 500_000.0
  rope_scaling: Llama31RopeScaling | None = None

  def num_params(self) -> int:
    e, f, d = self.emb_dim, self.mlp_dim, self.head_dim
    attn = e * d * (2 * self.num_query_heads + 2 * self.num_kv_heads)
    per_layer = attn + 3 * e * f + 2 * e
    return self.num_layers * per_layer + 2 * self.vocab_size * e + e

  def flops_per_token(self, seq_len: int) -> float:
    """Training FLOPs per token: 6N for weights plus causal attention."""
    non_embedding = self.num_params() - self.vocab_size * self.emb_dim
    attn = 6 * self.num_layers * self.num_query_heads * self.head_dim * seq_len
    return 6 * non_embedding + attn


FLAVORS = {
    "llama3.1-8b": LlamaConfig(
        vocab_size=128256, emb_dim=4096, num_layers=32, num_query_heads=32,
        num_kv_heads=8, head_dim=128, mlp_dim=14336,
        rope_scaling=Llama31RopeScaling(),
    ),
    "llama3-test": LlamaConfig(
        vocab_size=128256, emb_dim=256, num_layers=4, num_query_heads=8,
        num_kv_heads=2, head_dim=32, mlp_dim=512,
    ),
}


def _normal(key, shape, std):
  return std * jax.random.normal(key, shape, jnp.float32)


def stage_layer_counts(
    cfg: LlamaConfig, num_stages: int, layers: Sequence[int] | None = None
) -> tuple[int, ...]:
  """Decoder layers per stage: `layers` if given, else an even split."""
  if layers is None:
    if cfg.num_layers % num_stages:
      raise ValueError(f"{cfg.num_layers} layers do not split evenly into {num_stages} stages")
    return (cfg.num_layers // num_stages,) * num_stages
  if len(layers) != num_stages or sum(layers) != cfg.num_layers or min(layers) < 1:
    raise ValueError(f"stage layers {tuple(layers)} do not split {cfg.num_layers} layers into {num_stages} stages")
  return tuple(layers)


def init_stage_params(
    cfg: LlamaConfig,
    key: jax.Array,
    stage: int,
    num_stages: int,
    layers: Sequence[int] | None = None,
) -> dict:
  """Returns fp32 parameters for one pipeline stage.

  `layers` gives the decoder layers of each stage; the default splits them
  evenly. The last stage also holds the output head, so giving it fewer
  layers balances the stages.
  """
  n = stage_layer_counts(cfg, num_stages, layers)[stage]
  e, f, d = cfg.emb_dim, cfg.mlp_dim, cfg.head_dim
  hq, hkv = cfg.num_query_heads, cfg.num_kv_heads
  ks = jax.random.split(key, 9)
  out_std = 0.02 / (2 * cfg.num_layers) ** 0.5
  params = {
      "layers": {
          "attn_norm": jnp.ones((n, e), jnp.float32),
          "wq": _normal(ks[0], (n, e, hq * d), 0.02),
          "wk": _normal(ks[1], (n, e, hkv * d), 0.02),
          "wv": _normal(ks[2], (n, e, hkv * d), 0.02),
          "wo": _normal(ks[3], (n, hq * d, e), out_std),
          "mlp_norm": jnp.ones((n, e), jnp.float32),
          "w_gate": _normal(ks[4], (n, e, f), 0.02),
          "w_up": _normal(ks[5], (n, e, f), 0.02),
          "w_down": _normal(ks[6], (n, f, e), out_std),
      }
  }
  if stage == 0:
    params["embed"] = _normal(ks[7], (cfg.vocab_size, e), 0.02)
  if stage == num_stages - 1:
    params["final_norm"] = jnp.ones((e,), jnp.float32)
    params["lm_head"] = _normal(ks[8], (e, cfg.vocab_size), 0.02)
  return params


_LAYER_AXES = {
    "attn_norm": ("embed",),
    "wq": ("embed", "heads"),
    "wk": ("embed", "kv_heads"),
    "wv": ("embed", "kv_heads"),
    "wo": ("heads", "embed"),
    "mlp_norm": ("embed",),
    "w_gate": ("embed", "mlp"),
    "w_up": ("embed", "mlp"),
    "w_down": ("mlp", "embed"),
}
_OTHER_AXES = {
    "embed": ("vocab_in", "embed"),
    "final_norm": ("embed",),
    "lm_head": ("embed", "vocab"),
}


def logical_axes(path: tuple, leaf) -> tuple | None:
  """Logical axes of a parameter, or of an optimizer-state leaf mirroring one.

  Returns None for leaves that are not parameter-shaped, such as step counts.
  """
  keys = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
  if not keys:
    return None
  if "layers" in keys[:-1]:
    axes = ("layers",) + _LAYER_AXES.get(keys[-1], ())
  else:
    axes = _OTHER_AXES.get(keys[-1], ())
  return axes if len(axes) == getattr(leaf, "ndim", -1) else None


def device_copy_axes(params: dict) -> dict:
  """Axes to stack per-device parameter copies along, one per leaf.

  Layer weights take axis 1, after the layer axis, where `lax.scan` batches
  its inputs; other weights take axis 0.
  """
  return {k: jax.tree.map(lambda _, k=k: int(k == "layers"), v) for k, v in params.items()}


def merge_stage_params(stage_params: Sequence[dict]) -> dict:
  """Joins per-stage parameters into the parameters of a one-stage model."""
  return {
      "layers": jax.tree.map(
          lambda *xs: jnp.concatenate(xs), *(p["layers"] for p in stage_params)
      ),
      "embed": stage_params[0]["embed"],
      "final_norm": stage_params[-1]["final_norm"],
      "lm_head": stage_params[-1]["lm_head"],
  }


def _rope_freqs(cfg: LlamaConfig) -> jax.Array:
  dim = cfg.head_dim
  freqs = cfg.rope_theta ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
  s = cfg.rope_scaling
  if s is None:
    return freqs
  wavelen = 2 * jnp.pi / freqs
  smooth = (s.original_context / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor)
  blended = (1 - smooth) * freqs / s.factor + smooth * freqs
  scaled = jnp.where(wavelen > s.original_context / s.low_freq_factor, freqs / s.factor, blended)
  return jnp.where(wavelen < s.original_context / s.high_freq_factor, freqs, scaled)


def _rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
  """Rotary embedding on [batch, seq, heads, head_dim], split-halves layout."""
  seq = x.shape[1]
  angles = jnp.arange(seq, dtype=jnp.float32)[:, None] * freqs[None, :]
  sin = jnp.sin(angles)[None, :, None, :]
  cos = jnp.cos(angles)[None, :, None, :]
  x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
  out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
  return out.astype(x.dtype)


def _decoder_layer(cfg: LlamaConfig, h: jax.Array, p: dict) -> jax.Array:
  b, s, _ = h.shape
  x = ops.rms_norm(h, p["attn_norm"], cfg.norm_eps)
  q = (x @ p["wq"]).reshape(b, s, cfg.num_query_heads, cfg.head_dim)
  k = (x @ p["wk"]).reshape(b, s, cfg.num_kv_heads, cfg.head_dim)
  v = (x @ p["wv"]).reshape(b, s, cfg.num_kv_heads, cfg.head_dim)
  freqs = _rope_freqs(cfg)
  q, k = _rope(q, freqs), _rope(k, freqs)
  attn = ops.attention(q, k, v).reshape(b, s, cfg.num_query_heads * cfg.head_dim)
  h = h + attn @ p["wo"]
  x = ops.rms_norm(h, p["mlp_norm"], cfg.norm_eps)
  return h + ops.swiglu(x, p["w_gate"], p["w_up"], p["w_down"])


def _run_layers(
    cfg: LlamaConfig, layers: dict, h: jax.Array, remat: bool, compute_dtype
):
  def body(carry, p):
    return _decoder_layer(cfg, carry, p), None

  if remat:
    body = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
  # Casting inside the rematerialized body instead costs a second cast in
  # the backward pass: on MI355X, 2-3% slower and 5-10 GiB more per GPU.
  layers = jax.tree.map(lambda w: w.astype(compute_dtype), layers)
  h, _ = jax.lax.scan(body, h, layers)
  return h


def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
  logits = logits.astype(jnp.float32)
  logz = jax.nn.logsumexp(logits, axis=-1)
  gold = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
  return jnp.mean(logz - gold)


def stage_forward(
    params: dict,
    inputs: jax.Array,
    targets: jax.Array | None,
    *,
    cfg: LlamaConfig,
    stage: int,
    num_stages: int,
    remat: bool = True,
    constrain=None,
    compute_dtype=jnp.bfloat16,
) -> jax.Array:
  """Runs one stage: token ids in for stage 0, a loss out for the last stage.

  `constrain` places a sharding constraint on activations at stage boundaries.
  """
  if stage == 0:
    h = params["embed"][inputs].astype(compute_dtype)
  else:
    h = inputs
  if constrain is not None:
    h = constrain(h)
  h = _run_layers(cfg, params["layers"], h, remat, compute_dtype)
  if stage < num_stages - 1:
    return constrain(h) if constrain is not None else h
  h = ops.rms_norm(h, params["final_norm"], cfg.norm_eps)
  logits = jnp.einsum(
      "bse,ev->bsv", h, params["lm_head"].astype(compute_dtype),
      preferred_element_type=jnp.float32,
  )
  return cross_entropy(logits, targets)
