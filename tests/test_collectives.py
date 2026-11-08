"""RCCL and MORI collectives, the step model, time left and the performance database."""

import json
import pathlib

import pytest

from jax_titan.observability import perfdb
from jax_titan.planning import collectives
from jax_titan.planning import gpus
from jax_titan.planning import models
from jax_titan.planning import remaining
from jax_titan.planning import step

RESULTS = pathlib.Path(__file__).parent.parent / "docs" / "results"
MI355X = gpus.get("MI355X")

RCCL_TESTS_OUTPUT = """\
# rccl-tests version 2.18.3-:+ rccl-headers=23007 rccl-library=23007
# Collective test starting: all_reduce_perf
# nThread 2 nGpus 1 minBytes 1024 maxBytes 4096 step: 2(factor) warmup iters: 5 iters: 20
#  Rank  0 Group  0 Pid 1 on host-a device  0 [0000:05:00] AMD Instinct MI355X
#  Rank  1 Group  0 Pid 1 on host-b device  0 [0000:05:00] AMD Instinct MI355X
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong algo      proto      nchannels
        1024           256     float     sum      -1    10.08    0.10    0.10       0    10.00    0.10    0.10       0    RING        LL           1
        4096          1024     float     sum      -1     9.18    0.45    0.45       0     9.20    0.45    0.45     N/A     N/A       N/A         N/A
"""


def test_bus_bandwidth_factors():
  assert collectives.bus_factor("all_reduce", 8) == 1.75
  assert collectives.bus_factor("all_gather", 8) == collectives.bus_factor("all_to_all", 8) == 7 / 8
  assert collectives.bus_factor("send_recv", 2) == 1


def test_parse_rccl_tests():
  points = collectives.parse_rccl_tests(RCCL_TESTS_OUTPUT, "x.out")
  assert [(p.op, p.ranks, p.nodes, p.bytes) for p in points] == [
      ("all_reduce", 2, 2, 1024), ("all_reduce", 2, 2, 4096)]
  assert (points[0].algorithm, points[0].protocol, points[0].channels) == ("RING", "LL", 1)
  assert points[1].algorithm is None and points[0].gpu == "AMD Instinct MI355X"


def test_shipped_measurements_and_the_model_agree():
  table = collectives.measured()
  groups = collectives.groups(table.points)
  ar8 = next(g for g in groups if (g["op"], g["ranks"], g["forced"]) == ("all_reduce", 8, None))
  assert ar8["peak_busbw_gbs"] == pytest.approx(380, rel=0.02) and ar8["latency_us"] < 30
  for op, ranks in (("all_reduce", 8), ("all_gather", 8), ("send_recv", 2)):
    measured = collectives.predict(op, 2**28, ranks, gpu=MI355X)
    assert measured["source"].startswith("rccl_")
    assert collectives.model(op, 2**28, ranks, gpu=MI355X)["time_us"] == pytest.approx(
        measured["time_us"], rel=0.1)
  # Beyond the sweep, at the last measured bandwidth.
  big = collectives.predict("all_reduce", 2**34, 4, gpu=MI355X)
  assert big["busbw_gbs"] == pytest.approx(collectives.predict(
      "all_reduce", 2**32, 4, gpu=MI355X)["busbw_gbs"], rel=1e-6)


def test_rccl_model_across_nodes():
  small = collectives.model("all_reduce", 2**16, 1024, 128, gpu=MI355X)
  large = collectives.model("all_reduce", 2**30, 1024, 128, gpu=MI355X)
  assert small["algorithm"].startswith("tree") and large["algorithm"].startswith("ring")
  # A ring across nodes is bounded by the group's NICs: 8 x 50 GB/s x 0.9 here.
  assert large["busbw_gbs"] < 8 * 50 * collectives.NIC_EFFICIENCY
  one_rank_per_node = collectives.model("all_reduce", 2**30, 64, 64, gpu=MI355X)
  assert one_rank_per_node["busbw_gbs"] < 50 * collectives.NIC_EFFICIENCY


def test_mori_ep_matches_its_published_latency():
  for ranks, nodes, latency in ((8, 1, 31), (16, 2, 84), (32, 4, 152)):
    r = collectives.model("dispatch", 128 * 8 * 7168, ranks, nodes, gpu=MI355X, library="mori")
    assert r["time_us"] == pytest.approx(latency, rel=1e-6)
    assert r["algorithm"] == f"mori-ep{ranks}"
  mi350 = collectives.model("combine", 2**20, 8, gpu=gpus.get("MI350X"), library="mori")
  assert "MI355X" in mi350["source"]
  no_data = collectives.model("dispatch", 2**20, 8, gpu=gpus.get("MI455X"), library="mori")
  assert no_data["algorithm"].startswith("direct")


def _recorded_run():
  run = RESULTS / "2026-11-07" / "llama3.1-8b_pp2_run-dir"
  measured = step.measured_parts(run)
  return models.from_name("llama3.1-8b"), step.run_layout(measured["start"]), measured


def test_step_predicts_the_recorded_runs_collectives():
  model, lo, measured = _recorded_run()
  assert (lo.gpus, lo.pp, lo.dp, lo.schedule, lo.reduce_grads) == (8, 2, 4, "zb-h1", "step")
  r = step.predict(model, MI355X, lo)
  dp = next(c for c in r["collectives"] if c["kind"] == "dp")
  stage = (model.params() - model.embedding_params()) / 2 + model.vocab * model.emb
  assert dp["bytes"] == pytest.approx(4 * stage) and dp["calls"] == 1
  pp = next(c for c in r["collectives"] if c["kind"] == "pp")
  assert (pp["bytes"], pp["calls"]) == (2048 * 4096 * 2, 16)
  # The all-reduce of 15 GiB over 4 GPUs at RCCL's measured bandwidth, against the
  # profile's exposed collective time.
  assert sum(r["exposed_s"].values()) == pytest.approx(measured["collective"], rel=0.15)


def test_calibration_reproduces_the_run_and_scales():
  model, lo, measured = _recorded_run()
  c = step.calibrate(model, MI355X, lo, measured)
  assert c["bottleneck"] == "compute"
  assert c["fitted"]["step_s"] == pytest.approx(measured["step_s"], rel=0.01)
  fit = {"compute_efficiency": c["fit"]["compute_efficiency"], "overlap": c["fit"]["overlap"],
         "runtime_s": c["fit"]["runtime_s"]}
  rows = step.ladder(model, MI355X, lo, [8, 16, 64, 1024], batch_tokens=2**22, **fit)
  assert [r["gpus"] for r in rows] == [8, 16, 64]  # 1,024 GPUs: under a sequence per rank
  assert rows[1]["compute_s"] == pytest.approx(rows[0]["compute_s"] / 2)
  assert rows[2]["collectives"][0]["nodes"] == 4  # a stage's 32 GPUs
  weak = step.ladder(model, MI355X, lo, [8, 64], weak=True, **fit)
  assert weak[1]["compute_s"] == pytest.approx(weak[0]["compute_s"])


def test_expert_parallel_collectives_use_mori():
  mixtral = models.Model("mixtral", 32000, 4096, 32, 32, 8, 128, 14336, experts=8,
                         experts_per_token=2, expert_mlp=14336, moe_layers=32)
  lo = step.Layout(gpus=16, ep=8, microbatches=4, microbatch_size=16, seq_len=4096)
  r = step.predict(mixtral, MI355X, lo)
  ep = [c for c in r["collectives"] if c["kind"] == "ep"]
  assert [c["library"] for c in ep] == ["mori", "mori"]
  tokens = 16 * 4096 / lo.dp
  assert ep[0]["bytes"] == tokens * 2 * 4096 * 1 and ep[1]["bytes"] == tokens * 2 * 4096 * 2
  assert ep[0]["calls"] == 32 * 4 * 3


def _write_run(path, steps, tokens_per_step=1000, step_s=2.0, launch=0.0):
  lines = [{"event": "start", "time": launch + 5, "launch_time": launch, "args": {}},
           {"event": "layout", "time": launch + 6, "tokens_per_step": tokens_per_step}]
  t = launch + 10
  for s in steps:
    t += step_s
    lines.append({"step": s, "time": t, "step_time_s": step_s,
                  "tokens_per_s": tokens_per_step / step_s, "mfu": 0.1})
  path.mkdir(parents=True, exist_ok=True)
  (path / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_time_left(tmp_path):
  _write_run(tmp_path / "run", range(100))
  r = remaining.estimate(tmp_path / "run", target_tokens=1e6, window=50, now=1e9)
  assert r["tokens_done"] == 100 * 1000 and r["recent_tokens_per_s"] == 500
  productive = 100 * 2.0 / 210.0  # 200 s of steps after a 10 s startup
  assert r["productive_share"] == pytest.approx(productive)
  assert r["remaining_s"] == pytest.approx((1e6 - 1e5) / (500 * productive))


def test_perfdb_flags_a_regression(tmp_path):
  db = tmp_path / "perf.jsonl"
  key = {"model": "m", "gpu": "g", "gpus": 8}
  for t, s in ((1.0, 1.70), (2.0, 1.69), (3.0, 1.80)):
    perfdb.append(db, {"time": t, "run": f"r{t:g}", "key": key, "measured": {"step": s}})
  found = perfdb.regressions(perfdb.read(db), threshold=0.03)
  assert [g["record"]["run"] for g in found] == ["r3"] and found[0]["best"]["run"] == "r2"
