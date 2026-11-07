"""Semantic operators. Models call these instead of a kernel library.

Each op runs on the provider the current policy prefers (see
`jaxtitan.ops.registry`), and on `native` JAX by default:

  ops.set_policy("attention=te:aiter,linear=te_mxfp8")

Built-in providers: `native`; `te`, `te_mxfp8` and `te_fp8` (ROCm Transformer
Engine); `aiter` and `aiter_mxfp4` (JAX-AITER); `triton` (AITER's Triton
kernels) and `gluon` (gfx950 Gluon GEMMs), both through JAX-Triton. Packages
can add more under the `jaxtitan.ops_providers` entry-point group.
"""

import importlib

from jaxtitan.ops import native as _native
from jaxtitan.ops import registry
from jaxtitan.ops.registry import Impl
from jaxtitan.ops.registry import Policy
from jaxtitan.ops.registry import Provider
from jaxtitan.ops.registry import current_policy
from jaxtitan.ops.registry import for_stage
from jaxtitan.ops.registry import policy
from jaxtitan.ops.registry import register
from jaxtitan.ops.registry import reset_selections
from jaxtitan.ops.registry import selections
from jaxtitan.ops.registry import set_policy
from jaxtitan.ops.registry import summary


def _native_impls():
  return {
      "attention": Impl(_native.attention, "XLA"),
      "rms_norm": Impl(_native.rms_norm, "XLA"),
      "swiglu": Impl(_native.swiglu, "XLA"),
      "linear": Impl(_native.linear, "XLA"),
      "rope": Impl(_native.rope, "XLA"),
  }


def _lazy(module: str, factory: str):
  return lambda: getattr(importlib.import_module(module), factory)()


register(Provider(registry.NATIVE, _native_impls))
for _name, _module, _factory in (
    ("te", "jaxtitan.ops.providers.te", "bf16"),
    ("te_mxfp8", "jaxtitan.ops.providers.te", "mxfp8"),
    ("te_fp8", "jaxtitan.ops.providers.te", "fp8"),
    ("aiter", "jaxtitan.ops.providers.aiter", "bf16"),
    ("aiter_mxfp4", "jaxtitan.ops.providers.aiter", "mxfp4"),
    ("triton", "jaxtitan.ops.providers.triton", "aiter"),
    ("gluon", "jaxtitan.ops.providers.triton", "gluon"),
):
  register(Provider(_name, _lazy(_module, _factory)))


def attention(q, k, v, *, causal: bool = True, scale: float | None = None):
  """Attention over [batch, seq, heads, head_dim]; k and v may have fewer heads.

  `scale` defaults to head_dim ** -0.5.
  """
  return registry.dispatch("attention", q, k, v, causal=causal, scale=scale)


def rms_norm(x, weight, eps: float):
  return registry.dispatch("rms_norm", x, weight, eps)


def swiglu(x, w_gate, w_up, w_down):
  """silu(x @ w_gate) * (x @ w_up), times w_down."""
  return registry.dispatch("swiglu", x, w_gate, w_up, w_down)


def linear(x, w):
  """x [..., in] times w [in, out]."""
  return registry.dispatch("linear", x, w)


def rope(x, cos, sin):
  """Rotary embedding on [batch, seq, heads, head_dim], split-halves layout.

  `cos` and `sin` are [seq, head_dim // 2].
  """
  return registry.dispatch("rope", x, cos, sin)


__all__ = [
    "Impl", "Policy", "Provider", "attention", "current_policy", "for_stage", "linear",
    "policy", "register", "reset_selections", "rms_norm", "rope", "selections",
    "set_policy", "summary", "swiglu",
]
