"""Reference operator implementations in plain JAX."""

import jax
import jax.numpy as jnp


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
  x32 = x.astype(jnp.float32)
  y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
  return (y * weight.astype(jnp.float32)).astype(x.dtype)


def attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
  """Causal attention over [batch, seq, heads, head_dim]; k/v may have fewer heads."""
  return jax.nn.dot_product_attention(q, k, v, is_causal=True)


def swiglu(
    x: jax.Array, w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array
) -> jax.Array:
  return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down
