"""ROCm Transformer Engine ops.

`bf16()` gives fused attention and GEMMs. `mxfp8()` and `fp8()` give GEMMs
(`linear`, and the three in `swiglu`) quantized with TE's stateless recipes:
MXFP8 block scaling and per-tensor current scaling. TE's GEMMs and RMSNorm
have no working batching rule, so they decline under jax.vmap; with
gradients reduced per step they need `--per_device=shard_map`.
"""

import jax
import jax.numpy as jnp
import transformer_engine.jax  # noqa: F401  pylint: disable=unused-import
from transformer_engine.jax import attention as te_attention
from transformer_engine.jax import dense as te_dense
from transformer_engine.jax import layernorm as te_layernorm
from transformer_engine.jax import quantize as te_quantize
from transformer_engine.jax import sharding as te_sharding

from jax_titan.ops import registry
from jax_titan.ops.registry import Impl

_LAYOUT = te_attention.QKVLayout.BSHD_BSHD_BSHD
_HALF = (jnp.bfloat16, jnp.float16)

# TE's norm and GEMM partitioners require a process-wide MeshResource. None of
# its axes are named, so they partition from the arguments' shardings, which
# JAX-Titan's stage meshes set.
_SHARD_GUARD = te_sharding.global_shard_guard(te_sharding.MeshResource())
_SHARD_GUARD.__enter__()

_VMAP = ("TE's batching rule for it is missing or wrong, so it cannot run under jax.vmap; "
         "use --per_device=shard_map")
_under_vmap = registry.under_vmap


def _mask(causal):
  return te_attention.AttnMaskType.CAUSAL_MASK if causal else te_attention.AttnMaskType.NO_MASK


def attention(q, k, v, *, causal=True, scale=None):
  return te_attention.fused_attn(
      (q, k, v), None, None, None,
      attn_bias_type=te_attention.AttnBiasType.NO_BIAS,
      attn_mask_type=_mask(causal),
      qkv_layout=_LAYOUT,
      softmax_type=te_attention.AttnSoftmaxType.VANILLA_SOFTMAX,
      scaling_factor=q.shape[-1] ** -0.5 if scale is None else scale,
      dropout_probability=0.0,
      is_training=True,
      # TE's default "" fails its mesh-axis lookup once q, k, v are sharded.
      context_parallel_axis=None,
  )


def _attention_supports(q, k, v, *, causal=True, scale=None):
  del scale
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
    return "needs [batch, seq, heads, head_dim] q, k, v"
  if q.dtype not in _HALF or k.dtype != q.dtype or v.dtype != q.dtype:
    return f"needs bf16 or fp16 q, k, v, not {q.dtype}"
  b, sq, hq, d = q.shape
  sk, hkv = k.shape[1], k.shape[2]
  if hq % hkv:
    return f"{hq} query heads do not group over {hkv} kv heads"
  available = te_attention.is_fused_attn_kernel_available(
      True, b, q.dtype, k.dtype, _LAYOUT, te_attention.AttnBiasType.NO_BIAS,
      _mask(causal), te_attention.AttnSoftmaxType.VANILLA_SOFTMAX, 0.0,
      hq, hkv, sq, sk, d, v.shape[-1])
  return None if available else "no TE fused attention kernel for this shape"


def rms_norm(x, weight, eps):
  gamma = weight.astype(x.dtype)
  return te_layernorm.layernorm(x, gamma, jnp.zeros_like(gamma), "rmsnorm", epsilon=eps)


def _rms_norm_supports(x, weight, eps):
  del eps
  if _under_vmap(x, weight):
    return _VMAP
  if x.dtype not in (*_HALF, jnp.float32):
    return f"needs bf16, fp16 or fp32 x, not {x.dtype}"
  if weight.shape != (x.shape[-1],):
    return f"weight {weight.shape} does not match x {x.shape}"
  return None


def _gemm(scaling_mode):
  def linear(x, w):
    quantizers = (te_quantize.noop_quantizer_set if scaling_mode is None else
                  te_quantize.QuantizerFactory.create_set(
                      scaling_mode=scaling_mode, fwd_dtype=jnp.float8_e4m3fn,
                      bwd_dtype=jnp.float8_e5m2))
    return te_dense.dense(x, w.astype(x.dtype), contracting_dims=((x.ndim - 1,), (0,)),
                          quantizer_set=quantizers)
  return linear


def _gemm_supports(block):
  def supports(x, *ws):
    if _under_vmap(x, *ws):
      return _VMAP
    if x.dtype not in _HALF:
      return f"needs bf16 or fp16 x, not {x.dtype}"
    for w in ws:
      if w.ndim != 2:
        return "needs 2-D weights"
      if block and (w.shape[0] % block or w.shape[1] % block):
        return f"weight {w.shape} is not a multiple of {block}"
    rows = 1
    for n in x.shape[:-1]:
      rows *= n
    if block and rows % block:
      return f"{rows} rows are not a multiple of {block}"
    return None
  return supports


def _swiglu(linear):
  def swiglu(x, w_gate, w_up, w_down):
    return linear(jax.nn.silu(linear(x, w_gate)) * linear(x, w_up), w_down)
  return swiglu


def _require_gpu():
  platform = jax.devices()[0].platform
  if platform != "gpu":
    raise RuntimeError(f"TE runs on ROCm GPUs, not {platform}")


def bf16():
  """TE's BF16 ops.

  RMSNorm is not offered: with batch-sharded inputs under automatic
  partitioning its weight gradient is wrong (TE does not sum it over the
  data axis without a named MeshResource), and it is slower than XLA's.
  `rms_norm` stays here for the conformance harness.
  """
  _require_gpu()
  return {
      "attention": Impl(attention, "TE fused attention (AITER/CK FMHA)", _attention_supports),
      "linear": Impl(_gemm(None), "TE GEMM (hipBLASLt)", _gemm_supports(0)),
  }


def _quantized(mode, origin, block):
  _require_gpu()
  supported, reason = te_quantize.is_scaling_mode_supported(mode)
  if not supported:
    raise RuntimeError(f"{mode.name} is not supported here: {reason}")
  linear = _gemm(mode)
  return {
      "linear": Impl(linear, origin, _gemm_supports(block)),
      "swiglu": Impl(_swiglu(linear), origin, _gemm_supports(block)),
  }


def mxfp8():
  return _quantized(te_quantize.ScalingMode.MXFP8_1D_SCALING, "TE MXFP8 GEMM (hipBLASLt)", 32)


def fp8():
  return _quantized(te_quantize.ScalingMode.CURRENT_TENSOR_SCALING,
                    "TE FP8 current-scaling GEMM (hipBLASLt)", 16)
