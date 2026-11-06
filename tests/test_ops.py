"""Operator policy, dispatch and fallback, the conformance harness, and --ops."""

import json
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtitan import ops
from jaxtitan.ops import conformance
from jaxtitan.ops import registry

REPO = pathlib.Path(__file__).resolve().parents[1]


def _fake(supports=None):
  def rms_norm(x, weight, eps):
    return 2 * ops.native.rms_norm(x, weight, eps)
  return {"rms_norm": ops.Impl(rms_norm, "fake kernel", supports or registry._always)}


@pytest.fixture(autouse=True)
def _clean():
  registry.reset_selections()
  ops.set_policy("")
  yield
  ops.set_policy("")
  for name in [n for n in registry._PROVIDERS if n.startswith("fake")]:
    del registry._PROVIDERS[name]


def test_parse_policy():
  ops.register(ops.Provider("fake", _fake))
  p = ops.Policy.parse("attention=fake:native, rms_norm=fake,stage1.attention=native")
  assert p.prefs == {"attention": ("fake", "native"), "rms_norm": ("fake",)}
  assert p.stages == {1: {"attention": ("native",)}}
  assert p.for_stage(1).prefs["attention"] == ("native",)
  assert p.for_stage(0).prefs["attention"] == ("fake", "native")
  assert ops.Policy.parse(str(p)) == p
  for bad in ("attention", "attention=", "=native", "stagex.attention=native"):
    with pytest.raises(ValueError):
      ops.Policy.parse(bad)
  with pytest.raises(ValueError, match="unknown operator provider"):
    ops.Policy.parse("attention=nope")


def test_dispatch_prefers_and_records():
  ops.register(ops.Provider("fake", _fake))
  x, w = jnp.ones((2, 8)), jnp.ones((8,))
  ops.set_policy("rms_norm=fake")
  np.testing.assert_allclose(ops.rms_norm(x, w, 1e-6), 2 * ops.native.rms_norm(x, w, 1e-6))
  [s] = ops.selections()
  assert (s.op, s.provider, s.kernel_origin, s.skipped) == ("rms_norm", "fake", "fake kernel", ())
  assert s.signature == "float32[2, 8] float32[8]"
  assert "rms_norm -> fake (fake kernel)" in ops.summary()


def test_falls_back_with_reasons():
  ops.register(ops.Provider("fake_picky", lambda: _fake(lambda x, w, eps: "too picky")))

  def broken():
    raise ImportError("no such library")

  ops.register(ops.Provider("fake_broken", broken))
  ops.set_policy("rms_norm=fake_broken:fake_picky,attention=fake_picky")
  x, w = jnp.ones((2, 8)), jnp.ones((8,))
  np.testing.assert_allclose(ops.rms_norm(x, w, 1e-6), ops.native.rms_norm(x, w, 1e-6))
  q = jnp.ones((1, 4, 2, 8))
  ops.attention(q, q, q)
  by_op = {s.op: s for s in ops.selections()}
  assert by_op["rms_norm"].provider == "native"
  assert by_op["rms_norm"].skipped == (("fake_broken", "ImportError: no such library"),
                                       ("fake_picky", "too picky"))
  assert by_op["attention"].skipped == (("fake_picky", "has no attention"),)


def test_scoped_and_stage_policies():
  ops.register(ops.Provider("fake", _fake))
  x, w = jnp.ones((2, 8)), jnp.ones((8,))
  with ops.policy("rms_norm=fake"):
    ops.rms_norm(x, w, 1e-6)
  ops.rms_norm(x, w, 1e-6)
  assert [s.provider for s in ops.selections()] == ["fake", "native"]

  registry.reset_selections()
  ops.set_policy("rms_norm=fake,stage1.rms_norm=native")
  stage0 = ops.for_stage(lambda x: ops.rms_norm(x, w, 1e-6), 0)
  stage1 = ops.for_stage(lambda x: ops.rms_norm(x, w, 1e-6), 1)
  np.testing.assert_allclose(jax.jit(stage0)(x), 2 * jax.jit(stage1)(x))
  assert [s.provider for s in ops.selections()] == ["fake", "native"]


def test_conformance_native_on_cpu():
  rows = conformance.run(["native"], ["single", "vmap", "sharded"], quick=True)
  assert rows and all(r["status"] == "pass" for r in rows), rows


def test_trainer_records_ops(tmp_path):
  log = tmp_path / "log.jsonl"
  result = subprocess.run(
      [sys.executable, "-m", "jaxtitan.trainer.train", "--platform=cpu", "--model=llama3-test",
       "--data=synthetic", "--seq_len=64", "--microbatches=2", "--microbatch_size=4",
       "--steps=2", "--warmup_steps=1", "--ops=attention=te:native,stage1.rms_norm=native",
       f"--checkpoint_dir={tmp_path / 'ckpt'}", f"--log_file={log}"],
      capture_output=True, text=True, check=False, cwd=REPO,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  assert "attention -> native (XLA)" in result.stdout
  metas = [json.loads(p.read_text()) for p in (tmp_path / "ckpt" / "2" / "meta").iterdir()
           if p.is_file() and p.read_text().lstrip().startswith("{")]
  meta = next(m for m in metas if "ops" in m)
  assert {s["op"] for s in meta["ops"]} == {"attention", "rms_norm", "swiglu", "linear"}
