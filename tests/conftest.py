"""Tests run on 8 CPU devices; set up before anything imports jax."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402  pylint: disable=g-import-not-at-top
import pytest  # noqa: E402  pylint: disable=g-import-not-at-top

jax.config.update("jax_num_cpu_devices", 8)

# The ROCm runtime that jaxlib loads sets these with setenv(), outside
# os.environ. Subprocesses inherit them, force-load rocprofiler-sdk, and abort
# on a second LLVM ("Option 'bitcode-mdindex-threshold' registered more than
# once") when they import MaxText. Tests that start subprocesses after using
# jax themselves pass env=dict(os.environ), which never holds them.
_ROCPROFILER_VARS = ("ROCPROFILER_REGISTER_FORCE_LOAD", "ROCPROFILER_REGISTER_LIBRARY")


@pytest.fixture(autouse=True)
def _no_inherited_rocprofiler():
  for name in _ROCPROFILER_VARS:
    os.unsetenv(name)
  yield
  for name in _ROCPROFILER_VARS:
    os.unsetenv(name)
