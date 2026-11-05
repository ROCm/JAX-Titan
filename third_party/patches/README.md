# Patches

JAX-Titan applies these patches to its upstream dependencies until the changes
land upstream.

| Patch | Applies to | Change | Upstream |
|---|---|---|---|
| `jaxlib-ifrt-ir-on-pjrt.patch` | jax-ml/jax | Compile IFRT IR MPMD programs on PJRT-backed IFRT clients | |
| `shardy-mpmd-oss-mlir-import.patch` | openxla/shardy | Import MLIR from JAX in the MPMD JAX API | |

The XLA patch that accepted the ROCm platform in the IFRT IR atom program
passes is gone: it landed upstream as
[openxla/xla#52290](https://github.com/openxla/xla/pull/52290) (`dfca2c9`).

The patches contain and modify code from those projects, which are licensed
under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
They are provided under that license, not the MIT license that covers the rest
of JAX-Titan.
