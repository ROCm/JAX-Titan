# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import unittest
import pytest

import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.dlpack import to_dlpack
from parameterized import parameterized

from jaxpp.dlpack import capsule_name, dlpack_nccl_args

_GPU_RUNTIME_CANDIDATES = (
    ("libcudart.so", "cudaMemcpy"),
    ("libamdhip64.so", "hipMemcpy"),
)


def _load_gpu_runtime():
    for library_name, memcpy_name in _GPU_RUNTIME_CANDIDATES:
        try:
            library = ctypes.CDLL(library_name)
        except OSError:
            continue
        try:
            memcpy_fn = getattr(library, memcpy_name)
        except AttributeError:
            continue
        return library_name, memcpy_name, memcpy_fn
    return None, None, None


_library_name, _memcpy_name, _memcpy = _load_gpu_runtime()
if _memcpy is None:
    pytest.skip(
        "No CUDA/ROCm runtime found (libcudart.so or libamdhip64.so required)",
        allow_module_level=True,
    )

_memcpy.argtypes = [
    ctypes.c_void_p,  # dst
    ctypes.c_void_p,  # src
    ctypes.c_size_t,  # count (bytes)
    ctypes.c_int,  # kind
]
_memcpy.restype = ctypes.c_int
_MemcpyDeviceToHost = 2


def gpu_memcpy_to_host(device_ptr: int, num_bytes: int) -> bytes:
    host_buffer = (ctypes.c_uint8 * num_bytes)()
    err = _memcpy(host_buffer, device_ptr, num_bytes, _MemcpyDeviceToHost)
    if err != 0:
        raise RuntimeError(f"{_memcpy_name} failed with error {err}")
    return bytes(host_buffer)


class TestDlpackExport(unittest.TestCase):
    @parameterized.expand(
        [
            ("float32", jnp.float32, np.float32),
            ("bfloat16", jnp.bfloat16, ml_dtypes.bfloat16),
            ("float8_e4m3fn", jnp.float8_e4m3fn, ml_dtypes.float8_e4m3fn),
            ("float8_e5m2", jnp.float8_e5m2, ml_dtypes.float8_e5m2),
        ]
    )
    def test_dlpack_export(self, name, jax_dtype, np_dtype):
        x = jnp.array([1, 2, 3], dtype=jax_dtype)
        capsule = to_dlpack(x)
        self.assertEqual(capsule_name(capsule), "dltensor")
        data_ptr, count, nccl_dtype = dlpack_nccl_args(capsule)

        self.assertEqual(count, 3)

        itemsize = np.dtype(np_dtype).itemsize
        raw_bytes = gpu_memcpy_to_host(data_ptr, count * itemsize)
        values = np.frombuffer(raw_bytes, dtype=np_dtype)

        np.testing.assert_array_equal(values, np.array([1, 2, 3], dtype=np_dtype))

    def test_unsupported_dtype(self):
        x = jnp.array([1, 2, 3], dtype=jnp.float8_e4m3b11fnuz)
        capsule = to_dlpack(x)
        with self.assertRaises(ValueError) as ctx:
            dlpack_nccl_args(capsule)
        self.assertIn("Unsupported dtype", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
