"""Quantized GEMMs compiled by XLA, for `linear` and the three in `swiglu`.

`fp8()`: per-tensor current scaling; each operand is divided by its largest
magnitude over the largest E4M3 value and cast to E4M3, the incoming gradient
to E5M2. `mxfp8()` and `mxfp4()`: MaxText's `jax_mxfp8` and `jax_mxfp4`,
`jax.nn.scaled_dot_general` with E4M3 or E2M1 data in blocks of 32 along the
contracting dimension and an E8M0 scale per block, for the gradient too.
XLA runs every GEMM on hipBLASLt or on its Triton GEMMs.
"""

import math

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
def _fp8_dot(x, w):
  return _fp8_fwd(x, w)[0]


def _fp8_fwd(x, w):
  xq, sx = _quantize(x, jnp.float8_e4m3fn)
  wq, sw = _quantize(w, jnp.float8_e4m3fn)
  return _dot(xq, wq, ((1,), (0,)), sx * sw, x.dtype), (xq, sx, wq, sw)


def _fp8_bwd(residuals, g):
  xq, sx, wq, sw = residuals
  gq, sg = _quantize(g, jnp.float8_e5m2)
  return (_dot(gq, wq, ((1,), (1,)), sg * sw, g.dtype),
          _dot(xq, gq, ((0,), (0,)), sx * sg, g.dtype))


_fp8_dot.defvjp(_fp8_fwd, _fp8_bwd)


def _fp8_linear(x, w):
  y = _fp8_dot(x.reshape(-1, x.shape[-1]), w.astype(x.dtype))
  return y.reshape(*x.shape[:-1], w.shape[-1])


def _impls(linear, origin, block=1):
  def swiglu(x, w_gate, w_up, w_down):
    return linear(jax.nn.silu(linear(x, w_gate)) * linear(x, w_up), w_down)

  def supports(x, *ws):
    if x.dtype != jnp.bfloat16:
      return f"needs bf16 x, not {x.dtype}"
    if any(w.ndim != 2 for w in ws):
      return "needs 2-D weights"
    sizes = [math.prod(x.shape[:-1]), *(n for w in ws for n in w.shape)]
    if any(n % block for n in sizes):
      return f"needs rows and weight dimensions that are multiples of {block}, not {sizes}"
    return None

  return {"linear": Impl(linear, origin, supports), "swiglu": Impl(swiglu, origin, supports)}


def fp8():
  return _impls(_fp8_linear, "XLA FP8 current-scaling GEMM (hipBLASLt or Triton)")


def _mx(mode):
  config = jax.nn.get_scaled_dot_general_config(mode)

  def linear(x, w):
    return jax.nn.scaled_dot_general(
        x, w.astype(x.dtype), (((x.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=x.dtype, configs=[config] * 3)

  return _impls(linear, f"XLA {mode.upper()} scaled dot (hipBLASLt or Triton)", config.block_size)


def mxfp8():
  return _mx("mxfp8")


def mxfp4():
  return _mx("mxfp4")
