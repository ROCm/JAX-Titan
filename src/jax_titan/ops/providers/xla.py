"""Scalar FP8 GEMMs compiled by XLA, for `linear` and the three in `swiglu`.

Per-tensor current scaling: each operand is divided by its largest magnitude
over the largest E4M3 value and cast to E4M3, the incoming gradient to E5M2.
XLA runs every FP8 dot on hipBLASLt or on its Triton GEMMs.
"""

import jax
import jax.numpy as jnp

from jax_titan.ops.registry import Impl


def _quantize(x, dtype):
  fmax = float(jnp.finfo(dtype).max)
  amax = jnp.max(jnp.abs(x)).astype(jnp.float32)
  inverse = jnp.where(amax > 0, fmax / amax, 1.0)
  return jnp.clip(x.astype(jnp.float32) * inverse, -fmax, fmax).astype(dtype), 1 / inverse


def _dot(a, b, contract, scale, dtype):
  y = jax.lax.dot_general(a, b, (contract, ((), ())), preferred_element_type=dtype)
  return (y.astype(jnp.float32) * scale).astype(dtype)


@jax.custom_vjp
def _linear(x, w):
  return _linear_fwd(x, w)[0]


def _linear_fwd(x, w):
  xq, sx = _quantize(x, jnp.float8_e4m3fn)
  wq, sw = _quantize(w, jnp.float8_e4m3fn)
  return _dot(xq, wq, ((1,), (0,)), sx * sw, x.dtype), (xq, sx, wq, sw)


def _linear_bwd(residuals, g):
  xq, sx, wq, sw = residuals
  gq, sg = _quantize(g, jnp.float8_e5m2)
  return (_dot(gq, wq, ((1,), (1,)), sg * sw, g.dtype),
          _dot(xq, gq, ((0,), (0,)), sx * sg, g.dtype))


_linear.defvjp(_linear_fwd, _linear_bwd)


def linear(x, w):
  y = _linear(x.reshape(-1, x.shape[-1]), w.astype(x.dtype))
  return y.reshape(*x.shape[:-1], w.shape[-1])


def swiglu(x, w_gate, w_up, w_down):
  return linear(jax.nn.silu(linear(x, w_gate)) * linear(x, w_up), w_down)


def _supports(x, *ws):
  if x.dtype != jnp.bfloat16:
    return f"needs bf16 x, not {x.dtype}"
  if any(w.ndim != 2 for w in ws):
    return "needs 2-D weights"
  return None


def fp8():
  origin = "XLA FP8 current-scaling GEMM (hipBLASLt or Triton)"
  return {"linear": Impl(linear, origin, _supports), "swiglu": Impl(swiglu, origin, _supports)}
