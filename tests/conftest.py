"""Tests run on 8 CPU devices; set up before anything imports jax."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402  pylint: disable=g-import-not-at-top

jax.config.update("jax_num_cpu_devices", 8)
