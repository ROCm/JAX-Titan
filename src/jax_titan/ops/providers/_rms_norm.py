"""AITER's RMSNorm forward kernel, with its output pointers after its inputs.

jax-triton 0.5.0 hands a kernel its operand buffers and then its result
buffers, in that order, whatever the order of its pointer parameters, so a
result pointer between operands gets an operand's buffer. `_rms_norm_kernel`
takes its output pointer second.
"""

import triton
import triton.language as tl

from jax_titan.ops.providers import _kernel_sources

_rms_norm_kernel = _kernel_sources.aiter_kernels("normalization.rmsnorm")._rms_norm_kernel


@triton.jit
def rms_norm_fwd(input_ptr, g_ptr, output_ptr, rsigma_ptr, n_rows, n_cols, epsilon,
                 BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr, NUM_PRGMS: tl.constexpr):
  _rms_norm_kernel(input_ptr, output_ptr, g_ptr, rsigma_ptr, n_cols, n_cols, n_rows, n_cols,
                   epsilon, BLOCK_SIZE, USE_BLOCKED, NUM_PRGMS)
