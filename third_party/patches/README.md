# Patches

JAX-Titan applies these patches to its upstream dependencies until the changes
land upstream.

| Patch | Applies to | Change |
|---|---|---|
| `jaxlib-ifrt-ir-on-pjrt.patch` | jax-ml/jax | Compile IFRT IR MPMD programs on PJRT-backed IFRT clients |
| `xla-ifrt-ir-rocm-platform.patch` | openxla/xla | Accept the ROCm platform in the IFRT IR atom program passes |
| `shardy-mpmd-oss-mlir-import.patch` | openxla/shardy | Import MLIR from JAX in the MPMD JAX API |

The patches contain and modify code from those projects, which are licensed
under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
They are provided under that license, not the MIT license that covers the rest
of JAX-Titan.
