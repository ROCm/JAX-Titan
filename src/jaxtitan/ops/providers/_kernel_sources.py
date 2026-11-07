"""Imports the Triton and Gluon kernel sources of the `triton` provider.

The kernels come from submodules, without the PyTorch wrappers next to them:

- AITER's Triton kernels, `aiter.ops.triton._triton_kernels`, from the AITER
  that JAX-AITER pins (third_party/jax-aiter/third_party/aiter), or from
  JAXTITAN_AITER_ROOT. The `aiter` and `aiter.ops.triton` package inits import
  PyTorch, so unless an `aiter` package is installed they are created empty and
  only the kernel modules below them are imported.
- The gfx950 Gluon GEMMs of third_party/gfx950-gluon-tutorials, or of
  JAXTITAN_GLUON_TUTORIALS, whose kernel files import PyTorch for their host
  wrappers only.
"""

import contextlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import sys
import types

_REPO = pathlib.Path(__file__).resolve().parents[4]


def _root(env_var: str, default: pathlib.Path) -> pathlib.Path:
  root = pathlib.Path(os.environ.get(env_var) or default)
  if not root.is_dir():
    raise FileNotFoundError(
        f"{root} not found: check out the submodules (git submodule update --init "
        f"--recursive) or set {env_var}")
  return root


def _empty_package(name: str, path: pathlib.Path) -> None:
  spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
  spec.submodule_search_locations = [str(path)]
  sys.modules[name] = importlib.util.module_from_spec(spec)


def aiter_kernels(name: str) -> types.ModuleType:
  """`aiter.ops.triton._triton_kernels.<name>`, such as "normalization.rmsnorm"."""
  if "aiter" not in sys.modules and importlib.util.find_spec("aiter") is None:
    root = _root("JAXTITAN_AITER_ROOT", _REPO / "third_party/jax-aiter/third_party/aiter")
    for package in ("aiter", "aiter.ops", "aiter.ops.triton"):
      _empty_package(package, root.joinpath(*package.split(".")))
  return importlib.import_module(f"aiter.ops.triton._triton_kernels.{name}")


def aiter_config(path: str) -> dict:
  """A tuned config of AITER's, such as "gfx950/triton/fusions/silu_and_mul_backward/DEFAULT.json"."""
  aiter_kernels("common")
  configs = pathlib.Path(importlib.import_module("aiter.ops.triton").__path__[0]) / "configs"
  return json.loads((configs / path).read_text())


def _load(name: str, path: pathlib.Path) -> types.ModuleType:
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def _torch_placeholder() -> types.ModuleType:
  """A `torch` whose attributes are empty classes, for host code that is never run."""
  module = types.ModuleType("torch")
  module.__getattr__ = lambda name: type(name, (), {})
  return module


@contextlib.contextmanager
def _modules_while_importing(**modules):
  added = [name for name in modules if name not in sys.modules]
  for name in added:
    sys.modules[name] = modules[name]
  try:
    yield
  finally:
    for name in added:
      del sys.modules[name]


def gluon_gemm(kind: str) -> types.ModuleType:
  """The inter-wave Gluon GEMM module for `kind` ("a16w16", "a8w8" or "a4w4")."""
  name = f"jaxtitan_gluon_gemm_{kind}"
  if name in sys.modules:
    return sys.modules[name]
  gemm = _root("JAXTITAN_GLUON_TUTORIALS",
               _REPO / "third_party/gfx950-gluon-tutorials") / "kernels" / "gemm"
  common = _load("jaxtitan_gluon_gemm_common", gemm / "utils" / "common.py")
  # The kernel files import their shared helpers as `common`.
  with _modules_while_importing(common=common, torch=_torch_placeholder()):
    return _load(name, gemm / "inter_wave" / kind / "matmul_kernel.py")
