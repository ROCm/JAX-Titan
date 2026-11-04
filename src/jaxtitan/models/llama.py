"""Llama 3 decoder with parameters grouped by pipeline stage.

Architecture and flavors follow MaxText's `decoder_block: llama2` and its
`configs/models/llama3*.yml`: pre-norm RMSNorm, RoPE, grouped-query attention
and a SwiGLU MLP.
"""

from collections.abc import Sequence
import dataclasses
import functools

import jax
import jax.numpy as jnp

from jaxtitan import ops


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
    ),
    "llama3-test": LlamaConfig(
        vocab_size=128256, emb_dim=256, num_layers=4, num_query_heads=8,
        num_kv_heads=2, head_dim=32, mlp_dim=512,
    ),
}


def _normal(key, shape, std):
  return std * jax.random.normal(key, shape, jnp.float32)


def init_stage_params(
    cfg: LlamaConfig, key: jax.Array, stage: int, num_stages: int
) -> dict:
  """Returns fp32 parameters for one pipeline stage."""
  assert cfg.num_layers % num_stages == 0, (cfg.num_layers, num_stages)
  n = cfg.num_layers // num_stages
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


def _rope(x: jax.Array, theta: float) -> jax.Array:
  """Rotary embedding on [batch, seq, heads, head_dim], split-halves layout."""
  seq, dim = x.shape[1], x.shape[-1]
  freqs = theta ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
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
  q, k = _rope(q, cfg.rope_theta), _rope(k, cfg.rope_theta)
  attn = ops.attention(q, k, v).reshape(b, s, cfg.num_query_heads * cfg.head_dim)
  h = h + attn @ p["wo"]
  x = ops.rms_norm(h, p["mlp_norm"], cfg.norm_eps)
  return h + ops.swiglu(x, p["w_gate"], p["w_up"], p["w_down"])


def _run_layers(cfg: LlamaConfig, layers: dict, h: jax.Array, remat: bool):
  def body(carry, p):
    return _decoder_layer(cfg, carry, p), None

  if remat:
    body = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
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
  cast = functools.partial(jax.tree.map, lambda w: w.astype(compute_dtype))
  if stage == 0:
    h = cast(params["embed"])[inputs]
  else:
    h = inputs
  if constrain is not None:
    h = constrain(h)
  h = _run_layers(cfg, cast(params["layers"]), h, remat)
  if stage < num_stages - 1:
    return constrain(h) if constrain is not None else h
  h = ops.rms_norm(h, params["final_norm"], cfg.norm_eps)
  logits = jnp.einsum(
      "bse,ev->bsv", h, cast(params["lm_head"]),
      preferred_element_type=jnp.float32,
  )
  return cross_entropy(logits, targets)
