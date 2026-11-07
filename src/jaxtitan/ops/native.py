"""Reference operator implementations in plain JAX."""

import jax
import jax.numpy as jnp


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
  x32 = x.astype(jnp.float32)
  y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
  return (y * weight.astype(jnp.float32)).astype(x.dtype)


def attention(q: jax.Array, k: jax.Array, v: jax.Array, *, causal: bool = True,
              scale: float | None = None) -> jax.Array:
  """Attention over [batch, seq, heads, head_dim]; k/v may have fewer heads."""
  return jax.nn.dot_product_attention(q, k, v, scale=scale, is_causal=causal)


def swiglu(
    x: jax.Array, w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array
) -> jax.Array:
  return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down


def linear(x: jax.Array, w: jax.Array) -> jax.Array:
  """x [..., in] times w [in, out]."""
  return x @ w


def rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
  """Rotary embedding on [batch, seq, heads, head_dim], split-halves layout.

  `cos` and `sin` are [seq, head_dim // 2].
  """
  c = cos.astype(jnp.float32)[None, :, None, :]
  s = sin.astype(jnp.float32)[None, :, None, :]
  x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
  return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)
