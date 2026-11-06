"""Operator providers, the policy that picks one per op, and what it picked.

A provider is a named set of op implementations backed by a kernel library
(TE, JAX-AITER, ...), imported on first use. The policy lists, per op, the
providers to try in order; the first one that is installed and supports the
call wins, and `native` is always the last resort. The choice is made while
tracing, so it is fixed for a compiled program, and every choice is recorded
with the kernel library behind it.
"""

from collections.abc import Callable, Mapping
import contextlib
import contextvars
import dataclasses
import functools
import importlib.metadata
import threading

NATIVE = "native"
ENTRY_POINT_GROUP = "jaxtitan.ops_providers"


def _always(*args, **kwargs):
  del args, kwargs


def under_vmap(*args) -> bool:
  """Whether any argument is being traced by jax.vmap, under any other transform.

  Providers use it to decline kernels whose batching rules are missing or wrong.
  """
  for x in args:
    while x is not None:
      if type(x).__name__ == "BatchTracer":
        return True
      x = getattr(x, "primal", None)
  return False


@dataclasses.dataclass(frozen=True)
class Impl:
  """One provider's implementation of an op.

  `fn` takes the op's arguments. `supports(*args, **kwargs)` returns None when
  `fn` handles the call and otherwise the reason it does not. `kernel_origin`
  names the kernels behind `fn`, since providers wrap overlapping libraries.
  """

  fn: Callable
  kernel_origin: str
  supports: Callable[..., str | None] = _always


@dataclasses.dataclass
class Provider:
  """A named set of op implementations.

  `load()` imports the backing library and returns {op name: Impl}. It runs
  once; an exception marks the provider unavailable, with its message.
  """

  name: str
  load: Callable[[], Mapping[str, Impl]]
  _impls: Mapping[str, Impl] | None = dataclasses.field(default=None, repr=False)
  _error: str | None = dataclasses.field(default=None, repr=False)
  _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)

  def _load(self):
    with self._lock:
      if self._impls is None and self._error is None:
        try:
          self._impls = dict(self.load())
        except Exception as e:  # pylint: disable=broad-exception-caught
          self._error = f"{type(e).__name__}: {e}"

  def unavailable(self) -> str | None:
    """None if the provider loaded, else why it did not."""
    self._load()
    return self._error

  def impls(self) -> Mapping[str, Impl]:
    self._load()
    return self._impls or {}


_PROVIDERS: dict[str, Provider] = {}
_discovered = False


def register(provider: Provider) -> None:
  _PROVIDERS[provider.name] = provider


def _discover():
  global _discovered
  if _discovered:
    return
  _discovered = True
  for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
    if ep.name not in _PROVIDERS:
      register(Provider(ep.name, ep.load()))


def provider(name: str) -> Provider:
  _discover()
  if name not in _PROVIDERS:
    raise ValueError(f"unknown operator provider {name!r}; known: {', '.join(names())}")
  return _PROVIDERS[name]


def names() -> list[str]:
  _discover()
  return sorted(_PROVIDERS)


@dataclasses.dataclass(frozen=True)
class Policy:
  """Providers to try per op, in order, before `native`.

  `stages` holds per-stage preferences, which override `prefs` op by op.
  The spec form is comma-separated `op=provider[:provider...]` entries, with
  a `stageN.` prefix for one stage: `attention=te:aiter,stage1.attention=native`.
  """

  prefs: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
  stages: Mapping[int, Mapping[str, tuple[str, ...]]] = dataclasses.field(default_factory=dict)

  @classmethod
  def parse(cls, spec: str) -> "Policy":
    prefs, stages = {}, {}
    for entry in filter(None, (e.strip() for e in spec.split(","))):
      key, sep, value = entry.partition("=")
      providers = tuple(filter(None, value.split(":")))
      if not sep or not key or not providers:
        raise ValueError(f"bad ops policy entry {entry!r}; expected op=provider[:provider...]")
      for p in providers:
        provider(p)
      stage, dot, op = key.rpartition(".")
      if dot:
        if not stage.startswith("stage") or not stage[5:].isdigit():
          raise ValueError(f"bad ops policy entry {entry!r}; expected stageN.op=...")
        stages.setdefault(int(stage[5:]), {})[op] = providers
      else:
        prefs[op] = providers
    return cls(prefs, stages)

  def for_stage(self, stage: int) -> "Policy":
    return Policy({**self.prefs, **self.stages.get(stage, {})})

  def merged(self, other: "Policy") -> "Policy":
    """This policy with `other`'s preferences taking precedence, op by op."""
    stages = {s: {**self.stages.get(s, {}), **other.stages.get(s, {})}
              for s in {*self.stages, *other.stages}}
    return Policy({**self.prefs, **other.prefs}, stages)

  def __str__(self) -> str:
    entries = [f"{op}={':'.join(p)}" for op, p in sorted(self.prefs.items())]
    entries += [f"stage{s}.{op}={':'.join(p)}"
                for s, ops in sorted(self.stages.items()) for op, p in sorted(ops.items())]
    return ",".join(entries)


_global_policy = Policy()
_scoped_policy: contextvars.ContextVar[Policy | None] = contextvars.ContextVar(
    "jaxtitan_ops_policy", default=None)


def set_policy(policy: Policy | str) -> None:
  """Sets the process-wide policy."""
  global _global_policy
  _global_policy = Policy.parse(policy) if isinstance(policy, str) else policy


def current_policy() -> Policy:
  scoped = _scoped_policy.get()
  return _global_policy if scoped is None else scoped


@contextlib.contextmanager
def policy(override: Policy | str):
  """Applies `override` on top of the current policy while tracing inside."""
  override = Policy.parse(override) if isinstance(override, str) else override
  token = _scoped_policy.set(current_policy().merged(override))
  try:
    yield
  finally:
    _scoped_policy.reset(token)


def for_stage(fn: Callable, stage: int) -> Callable:
  """`fn` traced under the current policy's preferences for `stage`."""
  stage_policy = current_policy().for_stage(stage)

  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    token = _scoped_policy.set(stage_policy)
    try:
      return fn(*args, **kwargs)
    finally:
      _scoped_policy.reset(token)

  return wrapped


@dataclasses.dataclass(frozen=True)
class Selection:
  """One dispatch decision: the provider that ran `op` on `signature`.

  `skipped` lists the preferred providers passed over, with the reason.
  """

  op: str
  provider: str
  kernel_origin: str
  signature: str
  skipped: tuple[tuple[str, str], ...] = ()


_selections: dict[Selection, None] = {}


def _signature(args) -> str:
  parts = []
  for a in args:
    shape, dtype = getattr(a, "shape", None), getattr(a, "dtype", None)
    if shape is not None and dtype is not None:
      parts.append(f"{dtype}{list(shape)}")
  return " ".join(parts)


def dispatch(op: str, *args, **kwargs):
  """Runs `op` with the first preferred provider that can, else `native`."""
  order = current_policy().prefs.get(op, ())
  if NATIVE not in order:
    order = order + (NATIVE,)
  skipped = []
  for name in order:
    p = provider(name)
    reason = p.unavailable()
    impl = None
    if reason is None:
      impl = p.impls().get(op)
      reason = f"has no {op}" if impl is None else impl.supports(*args, **kwargs)
    if reason is None:
      _selections[Selection(op, name, impl.kernel_origin, _signature(args), tuple(skipped))] = None
      return impl.fn(*args, **kwargs)
    skipped.append((name, reason))
  raise RuntimeError(f"no provider ran {op}: {skipped}")


def selections() -> list[Selection]:
  """Every distinct dispatch decision since the last `reset_selections()`."""
  return list(_selections)


def reset_selections() -> None:
  _selections.clear()


def summary() -> str:
  """One line per op and provider, with the kernels and what was skipped."""
  lines = {}
  for s in _selections:
    skipped = "; ".join(f"{p} skipped: {r}" for p, r in s.skipped)
    line = f"{s.op} -> {s.provider} ({s.kernel_origin})" + (f" [{skipped}]" if skipped else "")
    lines[line] = None
  return "\n".join(lines) or "(no ops dispatched)"
