"""JAX-AITER ops, from the third_party/jax-aiter submodule.

`bf16()` gives flash attention, RMSNorm, the SiLU-and-mul of `swiglu` and BF16
GEMMs. `mxfp4()` gives MXFP4 GEMMs for `linear` and the three in `swiglu`.
AITER GEMMs compute A @ B^T, so weights [in, out] are transposed per call.
"""

import sys

import jax
import jax.numpy as jnp
from jax_aiter.activation import silu_and_mul
from jax_aiter.gemm import gemm
from jax_aiter.gemm_fp4 import gemm_fp4_bf16
from jax_aiter.mha import flash_attn_func
from jax_aiter.rmsnorm import rms_norm as aiter_rms_norm

from jax_titan.ops import registry
from jax_titan.ops.registry import Impl

_HALF = (jnp.bfloat16, jnp.float16)


def attention(q, k, v, *, causal=True, scale=None):
  out = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
  return out[0] if isinstance(out, tuple) else out


_under_vmap = registry.under_vmap
_VMAP = ("JAX-AITER has no batching rule for it (custom_partitioning), or a wrong one "
         "(RMSNorm), so it cannot run under jax.vmap; use --per_device=shard_map")


def _te_in_process() -> bool:
  policy = registry.current_policy()
  prefs = [*policy.prefs.values(), *(p for s in policy.stages.values() for p in s.values())]
  return "transformer_engine" in sys.modules or any(
      name.startswith("te") for providers in prefs for name in providers)


def _attention_supports(q, k, v, *, causal=True, scale=None):
  del causal, scale
  if _te_in_process():
    return ("its backward crashes in a process that also loads ROCm TE, which bundles "
            "its own AITER FMHA build")
  if _under_vmap(q, k, v):
    return _VMAP
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
    return "needs [batch, seq, heads, head_dim] q, k, v"
  if q.dtype not in _HALF or k.dtype != q.dtype or v.dtype != q.dtype:
    return f"needs bf16 or fp16 q, k, v, not {q.dtype}"
  if q.shape[2] % k.shape[2]:
    return f"{q.shape[2]} query heads do not group over {k.shape[2]} kv heads"
  if q.shape[-1] > 256:
    return f"head_dim {q.shape[-1]} is above 256"
  return None


def rms_norm(x, weight, eps):
  y = aiter_rms_norm(x.reshape(-1, x.shape[-1]), weight.astype(x.dtype), eps)
  return y.reshape(x.shape)


def _rms_norm_supports(x, weight, eps):
  del eps
  if _under_vmap(x, weight):
    return _VMAP
  if x.dtype not in _HALF:
    return f"needs bf16 or fp16 x, not {x.dtype}"
  if weight.shape != (x.shape[-1],):
    return f"weight {weight.shape} does not match x {x.shape}"
  return None


def _gemm(kernel):
  def linear(x, w):
    y = kernel(x.reshape(-1, x.shape[-1]), w.T.astype(x.dtype))
    return y.reshape(*x.shape[:-1], w.shape[-1])
  return linear


def _gemm_supports(k_multiple, n_multiple):
  def supports(x, *ws):
    if _under_vmap(x, *ws):
      return _VMAP
    if x.dtype != jnp.bfloat16:
      return f"needs bf16 x, not {x.dtype}"
    for w in ws:
      if w.ndim != 2:
        return "needs 2-D weights"
      if w.shape[0] % k_multiple or w.shape[1] % n_multiple:
        return f"weight {w.shape} is not a multiple of ({k_multiple}, {n_multiple})"
    return None
  return supports


def _swiglu(linear):
  def swiglu(x, w_gate, w_up, w_down):
    return linear(silu_and_mul(linear(x, w_gate), linear(x, w_up)), w_down)
  return swiglu


def _xla_linear(x, w):
  return x @ w


def _require_gpu():
  platform = jax.devices()[0].platform
  if platform != "gpu":
    raise RuntimeError(f"JAX-AITER runs on ROCm GPUs, not {platform}")


def bf16():
  _require_gpu()
  return {
      "attention": Impl(attention, "AITER flash attention (CK/ASM FMHA v3)", _attention_supports),
      "rms_norm": Impl(rms_norm, "AITER RMSNorm (CK)", _rms_norm_supports),
      "swiglu": Impl(_swiglu(_xla_linear), "AITER SiLU-and-mul with XLA GEMMs",
                     _gemm_supports(1, 1)),
      "linear": Impl(_gemm(gemm), "AITER BF16 ASM GEMM", _gemm_supports(64, 1)),
  }


def mxfp4():
  _require_gpu()
  linear = _gemm(gemm_fp4_bf16)
  return {
      "linear": Impl(linear, "AITER MXFP4 ASM GEMM", _gemm_supports(256, 32)),
      "swiglu": Impl(_swiglu(linear), "AITER MXFP4 ASM GEMMs and SiLU-and-mul",
                     _gemm_supports(256, 32)),
  }
