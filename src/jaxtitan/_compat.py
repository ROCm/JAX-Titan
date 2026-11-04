"""Import shims for running on stock jaxlib wheels."""

import importlib.util
import sys
import types


def _alias_internal_jaxlib_module() -> None:
  # Stock jaxlib's _sdy_mpmd imports `jax.jaxlib._jax`, which only exists in
  # Google's internal layout. JAX-Titan's jaxlib patch imports `jaxlib._jax`.
  if "jax.jaxlib._jax" in sys.modules:
    return
  if importlib.util.find_spec("jaxlib._jax") is None:
    return
  import jaxlib._jax  # pylint: disable=g-import-not-at-top

  alias = types.ModuleType("jax.jaxlib")
  alias._jax = jaxlib._jax  # pylint: disable=protected-access
  sys.modules.setdefault("jax.jaxlib", alias)
  sys.modules.setdefault("jax.jaxlib._jax", jaxlib._jax)


_alias_internal_jaxlib_module()
