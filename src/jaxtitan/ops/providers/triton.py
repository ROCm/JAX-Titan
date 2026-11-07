"""Triton and Gluon ops through jax-triton, from the third_party/jax-triton submodule.

`aiter()` gives AITER's Triton RMSNorm, SiLU-and-mul (for `swiglu`, after one
GEMM over the stacked gate and up weights) and RoPE kernels. `gluon()` gives
the gfx950 inter-wave Gluon BF16 GEMM of gfx950-gluon-tutorials for `linear`
and the three GEMMs of `swiglu`. `_kernel_sources` says where kernels come from.

jax-triton has no autodiff or sharding rules, so every kernel launch here is a
custom_partitioning op inside a custom_vjp.
"""

import functools
import importlib

import jax
import jax.numpy as jnp
import jax_triton as jt
from jax_triton import triton_lib
import triton

from jaxtitan.ops import registry
from jaxtitan.ops.partitioned import partitioned
from jaxtitan.ops.providers import _kernel_sources
from jaxtitan.ops.registry import Impl

_HALF = (jnp.bfloat16, jnp.float16)
_F32 = jnp.float32
_CUS = {"gfx942": 304, "gfx950": 256}
_VMAP = ("custom_partitioning has no batching rule, so it cannot run under jax.vmap; "
         "use --per_device=shard_map")


@functools.cache
def _arch() -> str:
  return triton_lib.triton_kernel_call_lib.get_arch_details(0).split(":")[0]


@functools.cache
def _num_cus() -> int:
  return getattr(jax.devices()[0], "core_count", None) or _CUS.get(_arch(), 256)


def _out(shape, dtype):
  return jax.ShapeDtypeStruct(shape, dtype)


# RMSNorm: AITER's forward, backward and weight-gradient reduction kernels.


def _norm_block(n: int, itemsize: int) -> tuple[int, bool]:
  pow2, cap = triton.next_power_of_2(n), 65536 // itemsize
  block = (8192 if pow2 > cap else pow2) if _arch() == "gfx950" else min(cap, pow2)
  return block, n > block


def _rms_norm_fwd_local(x, w, params):
  (eps,) = params
  m, n = x.shape
  block, blocked = _norm_block(n, x.dtype.itemsize)
  programs = min(m, _num_cus() * (4 if _arch() == "gfx950" and block <= 8192 else 1))
  return jt.triton_call(
      x, w, kernel=_rms_norm_kernels().rms_norm_fwd, n_rows=m, n_cols=n, epsilon=eps,
      BLOCK_SIZE=block, USE_BLOCKED=blocked, NUM_PRGMS=programs, grid=(programs,),
      out_type=(_out((m, n), x.dtype), _out((m,), _F32)))


def _rms_norm_bwd_local(dy, x, w, rsigma, params):
  del params
  m, n = x.shape
  block, blocked = _norm_block(n, x.dtype.itemsize)
  programs = min(m, _num_cus())
  partial_rows = m if blocked else programs
  dx, dw_partial = jt.triton_call(
      kernel=_rms()._rmsnorm_bwd_triton, grad_output_ptr=dy, input_ptr=x, g_ptr=w,
      rsigma_ptr=rsigma, input_row_stride=n, output_row_stride=n, n_rows=m, n_cols=n,
      BLOCK_SIZE=block, USE_BLOCKED=blocked, NUM_PRGMS=programs, num_warps=8,
      grid=(programs,), out_type=(_out((m, n), x.dtype), _out((partial_rows, n), _F32)))
  dw = jt.triton_call(
      kernel=_rms()._rmsnorm_bwd_dg_reduce_triton, dg_in_ptr=dw_partial, dg_in_stride=n,
      n_rows=partial_rows, n_cols=n, BLOCK_SIZE_M=128, BLOCK_SIZE_N=64,
      num_warps=8 if _arch() == "gfx950" else 4, grid=(triton.cdiv(n, 64),),
      out_type=_out((n,), _F32))
  return dx, dw


_rms_norm_fwd = partitioned(_rms_norm_fwd_local, (("r", "n"), ("n",)), (("r", "n"), ("r",)),
                             replicated=("n",))
_rms_norm_bwd = partitioned(_rms_norm_bwd_local, (("r", "n"), ("r", "n"), ("n",), ("r",)),
                             (("r", "n"), ("n",)), replicated=("n",))


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _rms_norm_2d(x, w, eps):
  return _rms_norm_fwd(x, w, (eps,))[0]


def _rms_norm_2d_fwd(x, w, eps):
  y, rsigma = _rms_norm_fwd(x, w, (eps,))
  return y, (x, w, rsigma)


def _rms_norm_2d_bwd(eps, residuals, dy):
  del eps
  x, w, rsigma = residuals
  dx, dw = _rms_norm_bwd(dy, x, w, rsigma, ())
  return dx, dw.astype(w.dtype)


_rms_norm_2d.defvjp(_rms_norm_2d_fwd, _rms_norm_2d_bwd)


def rms_norm(x, weight, eps):
  y = _rms_norm_2d(x.reshape(-1, x.shape[-1]), weight.astype(x.dtype), float(eps))
  return y.reshape(x.shape)


def _rms_norm_supports(x, weight, eps):
  del eps
  if registry.under_vmap(x, weight):
    return _VMAP
  if x.dtype not in _HALF:
    return f"needs bf16 or fp16 x, not {x.dtype}"
  if weight.shape != (x.shape[-1],):
    return f"weight {weight.shape} does not match x {x.shape}"
  return None


# SiLU-and-mul: AITER's kernels read [rows, 2, f], the gate then the up projection.


def _silu_mul_tiles(rows: int, f: int) -> tuple[int, int, int]:
  """AITER's forward tile picks for a [rows, f] output."""
  if f == 512:
    block_n = 512 if rows > 4096 else 256
  elif f == 384:
    block_n = 256 if rows <= 128 else 128
  else:
    block_n = max(32, 1 << (min(f, 1024).bit_length() - 1))
  if rows <= 64:
    block_m = min(32, max(4, triton.next_power_of_2(rows)))
  elif f == 384 and rows > 128:
    block_m = 32 if rows > 8192 else 8
  elif f == 512 and rows > 128 or block_n >= 512:
    block_m = 8
  else:
    block_m = 16
  num_warps = 8 if rows <= 128 and block_m >= 16 and block_n >= 128 else 2
  return block_m, block_n, num_warps


def _silu_mul_fwd_local(gu, params):
  del params
  rows, _, f = gu.shape
  block_m, block_n, num_warps = _silu_mul_tiles(rows, f)
  return jt.triton_call(
      kernel=_act().fused_silu_mul_kernel, inp_ptr=gu, n_rows=rows, n_cols=f,
      row_stride_in=2 * f, col_stride_in=1, row_stride_out=f, col_stride_out=1,
      BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps,
      grid=(triton.cdiv(rows, block_m), triton.cdiv(f, block_n)), out_type=_out((rows, f), gu.dtype))


@functools.cache
def _silu_mul_bwd_configs() -> dict:
  return _kernel_sources.aiter_config(
      f"{_arch()}/triton/fusions/silu_and_mul_backward/DEFAULT.json")


def _silu_mul_bwd_local(dout, gu, params):
  del params
  rows, _, f = gu.shape
  configs = _silu_mul_bwd_configs()
  bounds = sorted(int(k.removeprefix("N_LEQ_")) for k in configs if k.startswith("N_LEQ_"))
  config = dict(configs[next((f"N_LEQ_{b}" for b in bounds if f <= b), "any")])
  block_m = min(config.pop("BLOCK_M"), triton.next_power_of_2(rows))
  block_n = config.pop("BLOCK_N")
  return jt.triton_call(
      kernel=_act()._silu_and_mul_backward_kernel, grad_output_ptr=dout, input_ptr=gu,
      n_rows=rows, n_cols=f, grad_output_row_stride=f, grad_output_col_stride=1,
      input_row_stride=2 * f, input_col_stride=1, grad_input_row_stride=2 * f,
      grad_input_col_stride=1, BLOCK_M=block_m, BLOCK_N=block_n, **config,
      grid=(triton.cdiv(rows, block_m), triton.cdiv(f, block_n)), out_type=_out(gu.shape, gu.dtype))


_silu_mul_fwd = partitioned(_silu_mul_fwd_local, (("r", "c", "f"),), (("r", "f"),),
                             replicated=("c",))
_silu_mul_bwd = partitioned(_silu_mul_bwd_local, (("r", "f"), ("r", "c", "f")),
                             (("r", "c", "f"),), replicated=("c",))


@jax.custom_vjp
def _silu_mul(gu):
  return _silu_mul_fwd(gu, ())


def _silu_mul_vjp_fwd(gu):
  return _silu_mul(gu), gu


def _silu_mul_vjp_bwd(gu, dout):
  return (_silu_mul_bwd(dout, gu, ()),)


_silu_mul.defvjp(_silu_mul_vjp_fwd, _silu_mul_vjp_bwd)


def swiglu(x, w_gate, w_up, w_down):
  f = w_gate.shape[-1]
  w_gate_up = jnp.stack([w_gate, w_up], axis=1).astype(x.dtype)
  gu = jnp.einsum("...e,ecf->...cf", x, w_gate_up).reshape(-1, 2, f)
  h = _silu_mul(gu).reshape(*x.shape[:-1], f)
  return h @ w_down.astype(x.dtype)


def _swiglu_supports(x, w_gate, w_up, w_down):
  if registry.under_vmap(x, w_gate, w_up, w_down):
    return _VMAP
  if x.dtype not in _HALF:
    return f"needs bf16 or fp16 x, not {x.dtype}"
  if w_gate.ndim != 2 or w_up.shape != w_gate.shape:
    return f"needs matching 2-D gate and up weights, not {w_gate.shape} and {w_up.shape}"
  return None


# RoPE: AITER's kernels for cached cos and sin, on [batch, seq, heads, head_dim].


def _rope_local(kernel_name, x, cos, sin):
  b, s, h, d = x.shape
  strides = {"s": h * d, "b": s * h * d, "h": d, "d": 1}
  return jt.triton_call(
      kernel=getattr(_rope_kernels(), kernel_name), x_ptr=x, cos_ptr=cos, sin_ptr=sin,
      pos_ptr=None, off_ptr=None,
      **{f"stride_x_{k}": v for k, v in strides.items()},
      **{f"stride_out_{k}": v for k, v in strides.items()},
      stride_cos_s=d // 2, stride_cos_b=0, stride_cos_h=0, stride_cos_d=1,
      stride_pos_s=1, stride_pos_b=1, S=s, HAVE_NOPE=False, NOPE_FIRST=False,
      INPLACE=False, REUSE_FREQS_FRONT_PART=True, IS_NEOX=True, HAVE_POS=False,
      HAVE_OFFS=False, BLOCK_S=32, BLOCK_D=d, BLOCK_D_HALF=d // 2, num_warps=4,
      grid=(b, h, triton.cdiv(s, 32)), out_type=_out(x.shape, x.dtype))


_ROPE_DIMS = (("b", "s", "h", "d"), ("s", "e"), ("s", "e"))
_rope_fwd = partitioned(lambda x, c, s, p: _rope_local("_rope_kernel_sbhd_cached_fwd", x, c, s),
                         _ROPE_DIMS, (("b", "s", "h", "d"),), replicated=("d", "e"))
_rope_bwd = partitioned(lambda g, c, s, p: _rope_local("_rope_kernel_sbhd_cached_bwd", g, c, s),
                         _ROPE_DIMS, (("b", "s", "h", "d"),), replicated=("d", "e"))


@jax.custom_vjp
def _rope(x, cos, sin):
  return _rope_fwd(x, cos, sin, ())


def _rope_vjp_fwd(x, cos, sin):
  return _rope(x, cos, sin), (x, cos, sin)


def _rope_vjp_bwd(residuals, dy):
  x, cos, sin = residuals
  x1, x2 = jnp.split(x.astype(_F32), 2, axis=-1)
  g1, g2 = jnp.split(dy.astype(_F32), 2, axis=-1)
  dcos = jnp.sum(g1 * x1 + g2 * x2, axis=(0, 2))
  dsin = jnp.sum(g2 * x1 - g1 * x2, axis=(0, 2))
  return _rope_bwd(dy, cos, sin, ()), dcos, dsin


_rope.defvjp(_rope_vjp_fwd, _rope_vjp_bwd)


def rope(x, cos, sin):
  return _rope(x, cos.astype(_F32), sin.astype(_F32))


def _rope_supports(x, cos, sin):
  if registry.under_vmap(x, cos, sin):
    return _VMAP
  if x.ndim != 4 or x.dtype not in _HALF:
    return f"needs bf16 or fp16 [batch, seq, heads, head_dim] x, not {x.dtype}{list(x.shape)}"
  d = x.shape[-1]
  if d & (d - 1):
    return f"head_dim {d} is not a power of 2"
  if cos.shape != (x.shape[1], d // 2) or sin.shape != cos.shape:
    return f"needs [seq, head_dim // 2] cos and sin, not {cos.shape} and {sin.shape}"
  return None


# Gluon GEMM: a [m, k] times b_t [n, k] transposed, 256x256x64 tiles, two waves per SIMD.

_GEMM = {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "WARPS_M": 2, "WARPS_N": 4,
         "NUM_XCDS": 8, "GROUP_SIZE_M": 4}
_GEMM_OPTIONS = {"llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"),)}


def _gemm_fits(m: int, n: int, k: int) -> bool:
  """Its stores are unmasked, and its K loop is unrolled twice."""
  return m % 256 == 0 and n % 256 == 0 and k % 128 == 0 and k >= 256


def _gemm_local(a, b_t, params):
  del params
  (m, k), n = a.shape, b_t.shape[0]
  if not _gemm_fits(m, n, k):
    return jax.lax.dot_general(a, b_t, (((1,), (1,)), ((), ())))
  grid_mn = (m // 256) * (n // 256)
  return jt.triton_call(
      kernel=_gluon_gemm().a16w16, a_ptr=a, b_ptr=b_t, M=m, N=n, K=k, GRID_MN=grid_mn,
      **_GEMM, num_warps=8, backend_options=_GEMM_OPTIONS, grid=(grid_mn,),
      out_type=_out((m, n), a.dtype))


_gemm = partitioned(_gemm_local, (("r", "k"), ("c", "k")), (("r", "c"),))


@jax.custom_vjp
def _linear_2d(x, w):
  return _gemm(x, w.T, ())


def _linear_2d_fwd(x, w):
  return _linear_2d(x, w), (x, w)


def _linear_2d_bwd(residuals, dy):
  x, w = residuals
  return _gemm(dy, w, ()), _gemm(x.T, dy.T, ())


_linear_2d.defvjp(_linear_2d_fwd, _linear_2d_bwd)


def linear(x, w):
  y = _linear_2d(x.reshape(-1, x.shape[-1]), w.astype(x.dtype))
  return y.reshape(*x.shape[:-1], w.shape[-1])


def _linear_supports(x, *ws):
  if registry.under_vmap(x, *ws):
    return _VMAP
  if x.dtype != jnp.bfloat16:
    return f"needs bf16 x, not {x.dtype}"
  rows = x.size // x.shape[-1]
  for w in ws:
    if w.ndim != 2:
      return "needs 2-D weights"
    if not (_gemm_fits(rows, w.shape[1], w.shape[0]) and _gemm_fits(rows, w.shape[0], w.shape[1])):
      return f"{rows} rows by weight {w.shape} is not in 256x256 tiles with K a multiple of 128"
  return None


def gluon_swiglu(x, w_gate, w_up, w_down):
  return linear(jax.nn.silu(linear(x, w_gate)) * linear(x, w_up), w_down)


def _rms():
  return _kernel_sources.aiter_kernels("normalization.rmsnorm")


def _rms_norm_kernels():
  return importlib.import_module("jaxtitan.ops.providers._rms_norm")


def _act():
  return _kernel_sources.aiter_kernels("fusions.activation")


def _rope_kernels():
  return _kernel_sources.aiter_kernels("rope.rope")


def _gluon_gemm():
  return importlib.import_module("jaxtitan.ops.providers._gluon_gemm")


def _require_rocm():
  platform = jax.devices()[0].platform
  if platform != "gpu":
    raise RuntimeError(f"jax-triton kernels here run on ROCm GPUs, not {platform}")


def aiter():
  _require_rocm()
  _rms(), _rms_norm_kernels(), _act(), _rope_kernels()
  return {
      "rms_norm": Impl(rms_norm, "AITER Triton RMSNorm", _rms_norm_supports),
      "swiglu": Impl(swiglu, "AITER Triton SiLU-and-mul with XLA GEMMs", _swiglu_supports),
      "rope": Impl(rope, "AITER Triton RoPE", _rope_supports),
  }


def gluon():
  _require_rocm()
  if _arch() != "gfx950":
    raise RuntimeError(f"the Gluon GEMM is written for gfx950, not {_arch()}")
  _gluon_gemm()
  return {
      "linear": Impl(linear, "Gluon inter-wave BF16 GEMM", _linear_supports),
      "swiglu": Impl(gluon_swiglu, "Gluon inter-wave BF16 GEMMs with XLA SiLU-and-mul",
                     _linear_supports),
  }
