"""Training metrics: summaries, ETTR, Gate 0, collectives and power, from
synthetic logs with known answers, and the trainer's run directory."""

import json
import math
import os
import pathlib
import subprocess
import sys

import pytest

from jax_titan.observability import collectives
from jax_titan.observability import ettr
from jax_titan.observability import power
from jax_titan.observability import records
from jax_titan.observability import summarize

REPO = pathlib.Path(__file__).resolve().parents[1]
STEP_S, TOKENS = 2.0, 1000


def _attempt(t, first_step, last_step, saves, *, restored=0, completed=False, slow_after_save=1.0):
  """Records of one attempt starting at wall time `t`; returns (records, end time)."""
  out = [
      {"event": "start", "time": t + 3, "launch_time": t, "process_start": t + 1,
       "main_start": t + 2, "jax_ready": t + 3, "num_devices": 8, "process_count": 1,
       "device_kind": "test", "args": {"model": "m", "microbatches": 8, "seed": 0,
                                       "peak_tflops_per_device": 1.0}},
      {"event": "layout", "time": t + 4, "tokens_per_step": TOKENS, "flops_per_step": 8e12,
       "hardware_flops_per_step": 10e12},
      {"event": "state", "time": t + 10, "seconds": 5.0, "restored_step": restored},
      {"event": "compile", "time": t + 30, "lower_s": 1.0, "compile_s": 19.0},
      {"event": "data", "time": t + 31, "seconds": 1.0},
  ]
  t += 31
  pending = 0.0
  for step in range(first_step, last_step + 1):
    dt = STEP_S + (10.0 if step == first_step else 0.0) + pending
    pending = 0.0
    t += dt
    out.append({"step": step, "loss": 10.0 - step * 0.1, "grad_norm": 1.0, "step_time_s": dt,
                "tokens_per_s": TOKENS / dt, "mfu": 0.5, "time": t})
    if step + 1 in saves:
      t += 3.0
      out.append({"event": "save", "time": t, "step": step + 1, "blocking_s": 3.0})
      pending = slow_after_save
  if completed:
    out.append({"event": "end", "time": t, "status": "completed", "steps_done": last_step + 1 - first_step})
  return out, t


@pytest.fixture
def restarted_log(tmp_path):
  """Steps 0-11 with a save after step 5, a crash, then steps 6-15 from that save."""
  first, t = _attempt(0.0, 0, 11, {6})
  second, _ = _attempt(t + 20.0, 6, 15, {12, 16}, restored=6, completed=True)
  path = tmp_path / "train.jsonl"
  path.write_text("".join(json.dumps(r) + "\n" for r in first + second))
  return path


def test_summary_of_a_restarted_run(restarted_log):
  m = summarize.run_metrics(restarted_log, skip=2, wcp_steps=1)
  assert m["steps"]["attempts"] == 2
  assert m["step_time_s"]["p50"] == STEP_S and m["step_time_s"]["max"] == STEP_S
  assert m["tokens_per_s"] == TOKENS / STEP_S and m["tokens_per_s_per_gpu"] == TOKENS / STEP_S / 8
  assert m["mfu"] == pytest.approx(8e12 / STEP_S / 8e12)
  s = m["startup_s"]
  assert (s["launch"], s["python_imports"], s["jax_init"], s["init"], s["compile"], s["data"]) == (
      1, 1, 1, 5, 20, 1)
  assert s["first_step_excess"] == 10.0
  assert s["total"] == pytest.approx(31 + 12 - STEP_S)
  c = m["checkpoint"]
  assert c["saves"] == 3 and c["blocking_s"]["p50"] == 3.0
  assert c["w_cp_s"]["p50"] == 4.0
  assert c["restore_s"]["p50"] == 5.0


def test_mfu_variants_of_a_restarted_run(restarted_log):
  m = summarize.run_metrics(restarted_log, skip=2, wcp_steps=1)
  # 8e12 FLOPs per step on 8 devices of 1 TFLOP/s: MFU 0.5 at the 2 s steady step.
  assert m["mfu_sustained"] == pytest.approx(0.5) and m["hfu_sustained"] == pytest.approx(0.625)
  steps = records.steps(records.read(restarted_log))
  active = sum(r["step_time_s"] for r in steps)
  assert m["mfu_training_active"] == pytest.approx(len(steps) * STEP_S / active * 0.5)
  e = ettr.measure(restarted_log, skip=2, wcp_steps=1)
  assert m["mfu_end_to_end"] == pytest.approx(16 * 8e12 / e["wall_s"] / 8e12)
  assert m["mfu_end_to_end"] == pytest.approx(e["mfu_end_to_end"])
  rolling = summarize.rolling_mfu(records.read(restarted_log), 8e12, 8e12, window=4)
  # Past each attempt's first step, a window holds at most one step slowed by a save.
  assert min(r for _, _, r in rolling) == pytest.approx(4 * 8e12 / (4 * STEP_S + 1.0) / 8e12)
  assert max(r for _, _, r in rolling) == pytest.approx(0.5)


def test_aggregate_mfu():
  runs = [{"mfu_training_active": 0.5, "peak_flops_per_s": 1.0, "active_s": 30.0},
          {"mfu_training_active": 0.2, "peak_flops_per_s": 1.0, "active_s": 10.0}]
  assert summarize.aggregate_mfu(runs) == pytest.approx((0.5 * 30 + 0.2 * 10) / 40)


def test_flops_model_matches_the_trainer_and_balances_stages():
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top
  from jax_titan.observability import flops  # pylint: disable=g-import-not-at-top

  cfg = llama.FLAVORS["llama3.1-8b"]
  r = flops.report("llama3.1-8b", num_stages=2, seq_len=2048)
  assert r["model_flops_per_token"] == pytest.approx(cfg.flops_per_token(2048), rel=1e-4)
  layer = sum(flops.layer_forward(flops.Dims.of(cfg), 2048).values())
  recompute = cfg.num_layers * layer * r["tokens_per_step"]
  assert r["executed_flops"] - r["model_flops"] == pytest.approx(recompute)
  assert flops.recompute_flops_per_token(cfg, 2048, True) * r["tokens_per_step"] == recompute
  # The last stage also holds the output head.
  assert r["balanced_split"] == [17, 15]
  stage = r["stages"][0]
  assert stage["gradient_bytes"] == pytest.approx(2 * 3 / 4 * stage["params"] * 4)
  per_microbatch = flops.report("llama3.1-8b", reduce_grads="microbatch")["stages"][0]
  assert per_microbatch["gradient_bytes"] == pytest.approx(8 * stage["gradient_bytes"])


def test_bench_speedups():
  from jax_titan.ops import conformance  # pylint: disable=g-import-not-at-top

  rows = [{"mode": "bench", "status": "pass", "provider": "p", "op": op, "ms": ms,
           "native_ms": 2.0, "speedup": 2.0 / ms} for op, ms in (("a", 1.0), ("b", 4.0))]
  total = next(s for s in conformance.speedups(rows) if s["op"] == "all")
  assert total["geomean"] == pytest.approx(1.0) and total["time_weighted"] == pytest.approx(0.8)


def test_figures(restarted_log, tmp_path):
  pytest.importorskip("matplotlib")
  from jax_titan.observability import plot  # pylint: disable=g-import-not-at-top

  written = plot.plot_run(plot.series(restarted_log, window=4), tmp_path)
  assert written and written[0].stat().st_size > 10_000


def test_ettr_of_a_restarted_run(restarted_log):
  m = ettr.measure(restarted_log, skip=2, wcp_steps=1)
  assert m["committed_steps"] == 16 and m["lost_steps"] == 6
  assert m["productive_s"] == 16 * STEP_S
  assert m["losses"]["lost_work_s"] == pytest.approx(6 * STEP_S + 1.0)
  assert m["productive_s"] + sum(m["losses"].values()) == pytest.approx(m["wall_s"])
  assert m["ettr"] == pytest.approx(m["productive_s"] / m["wall_s"])
  assert m["effective_tokens_per_s"] == pytest.approx(m["ettr"] * TOKENS / STEP_S)


def test_daly_interval_and_projection():
  mttf = 25 * 86400.0
  tau = ettr.daly_interval(60.0, mttf)
  assert math.sqrt(2 * 60 * mttf) - 60 < tau < math.sqrt(2 * 60 * mttf) * 1.01
  rows = ettr.project(60.0, 300.0, [1, 8, 1024], [5.0])
  assert [r["nodes"] for r in rows] == [1, 8, 1024]
  assert rows[1]["mttf_h"] == pytest.approx(25 * 24)
  assert rows[0]["ettr"] > rows[1]["ettr"] > rows[2]["ettr"] > 0


def _run_log(path, step_times, losses=None, **args):
  start = {"event": "start", "time": 0, "main_start": 0, "jax_ready": 0,
           "num_devices": args.pop("devices", 8), "args": {"model": "m", **args}}
  rows = [start, {"event": "layout", "time": 0, "tokens_per_step": args.get("tokens", TOKENS)}]
  for i, dt in enumerate(step_times):
    rows.append({"step": i, "loss": (losses or [1.0] * len(step_times))[i], "grad_norm": 1.0,
                 "step_time_s": dt, "tokens_per_s": TOKENS / dt, "mfu": 0.1, "time": i + 1.0})
  if "hbm" in args:
    rows.append({"event": "memory", "time": 0, "peak_hbm_gib": [args["hbm"]] * 8})
  path.write_text("".join(json.dumps(r) + "\n" for r in rows))
  return path


def test_cross_run_reports(tmp_path):
  runs = [summarize.run_metrics(_run_log(tmp_path / f"{m}.jsonl", [0.5 + 0.1 * m] * 8,
                                         microbatches=m, hbm=100.0 + m), skip=1)
          for m in (4, 8)]
  fit = summarize.bubble(runs)
  assert fit[0]["fixed_s"] == pytest.approx(0.5) and fit[0]["per_microbatch_s"] == pytest.approx(0.1)
  assert fit[0]["utilization"]["8"] == pytest.approx(1 - 0.5 / 1.3)
  assert len(summarize.pareto(runs)) == 1
  scaled = [summarize.run_metrics(_run_log(tmp_path / f"n{n}.jsonl", [1.0 if n == 8 else 0.6] * 8,
                                           devices=n), skip=1) for n in (8, 16)]
  assert summarize.scaling(scaled)[1]["efficiency"] == pytest.approx((1 / 0.6) / 2)
  repeats = [summarize.run_metrics(_run_log(tmp_path / f"r{i}.jsonl", [1.0 + 0.01 * i] * 8, seed=i),
                                   skip=1) for i in range(3)]
  r = summarize.repeats(repeats)[0]
  assert r["median_s"] == pytest.approx(1.01) and r["regression_threshold_s"] > 1.02


def test_gate0(tmp_path):
  ref = _run_log(tmp_path / "ref.jsonl", [1.0] * 12, losses=[5.0] * 12)
  close = _run_log(tmp_path / "close.jsonl", [1.0] * 12, losses=[5.0 * (1 + 1e-4)] * 12)
  late = _run_log(tmp_path / "late.jsonl", [1.0] * 12, losses=[5.0] * 10 + [6.0, 6.0])
  passed, diverged = summarize.gate0(ref, [close, late], steps=10, rtol=1e-3)
  assert passed["pass"] and passed["max_rel_loss"] == pytest.approx(1e-4)
  assert diverged["pass"]
  assert not summarize.gate0(ref, [late], steps=12, rtol=1e-3)[0]["pass"]


def test_collectives_from_inspector_output(tmp_path):
  def obj(op, us, rank):
    return {"header": {"id": "0x1", "rank": rank, "n_ranks": 4, "nnodes": 1},
            "metadata": {"inspector_output_format_version": "v4.0"},
            "coll_perf": {"coll": op, "coll_sn": 1, "coll_msg_size_bytes": 1 << 20,
                          "coll_exec_time_us": us, "coll_busbw_gbs": 100.0,
                          "coll_timing_source": "kernel_gpu"}}
  (tmp_path / "rank0.log").write_text(
      "\n".join(json.dumps(obj("AllReduce", us, 0)) for us in range(1, 101)))
  (tmp_path / "rank1.log").write_text(json.dumps(obj("AllGather", 7.0, 1), indent=2))
  rows = {r["op"]: r for r in collectives.summarize(tmp_path)}
  assert rows["AllReduce"]["count"] == 100
  assert rows["AllReduce"]["time_us_p50"] == pytest.approx(50.5)
  assert rows["AllReduce"]["time_us_p99"] == pytest.approx(99.01)
  assert rows["AllGather"]["time_us_p99"] == 7.0


def test_energy_between():
  samples = [{"time": float(t), "gpu": "g0", "energy_j": 100.0 * t, "gfx_mhz": 2000, "temp_c": 50}
             for t in range(11)]
  e = power.energy_between(samples, 2.0, 8.0)["g0"]
  assert e["energy_j"] == 600.0 and e["mean_power_w"] == 100.0


def test_restart_rehearsal(tmp_path):
  run = tmp_path / "run"
  common = [sys.executable, "-m", "jax_titan.trainer.train", "--platform=cpu",
            "--model=llama3-test", "--data=synthetic", "--seq_len=32", "--microbatches=4",
            "--microbatch_size=4", "--steps=8", "--warmup_steps=1", "--checkpoint_every=3",
            f"--checkpoint_dir={tmp_path / 'ckpt'}", f"--run_dir={run}", "--power_sample_s=0"]
  first = subprocess.run([*common, "--fail_at_step=5"], capture_output=True, text=True,
                         check=False, cwd=REPO, env=dict(os.environ))
  assert first.returncode == 75, first.stdout[-2000:] + first.stderr[-2000:]
  second = subprocess.run(common, capture_output=True, text=True, check=False, cwd=REPO,
                          env=dict(os.environ))
  assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
  assert "restored step 3" in second.stdout
  m = ettr.measure(run, skip=1)
  assert (m["attempts"], m["committed_steps"], m["lost_steps"]) == (2, 8, 2)
  assert m["u0_stages_s"][1]["restore"] > 0


def test_trainer_run_dir(tmp_path):
  run = tmp_path / "run"
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", "--platform=cpu", "--model=llama3-test",
       "--data=synthetic", "--seq_len=32", "--microbatches=4", "--microbatch_size=4",
       "--steps=6", "--warmup_steps=1", "--eval_every=3", "--eval_batches=2",
       "--checkpoint_every=3", f"--checkpoint_dir={tmp_path / 'ckpt'}", f"--run_dir={run}",
       "--power_sample_s=0"],
      capture_output=True, text=True, check=False, cwd=REPO,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  log = records.read(run / "train.jsonl")
  names = [r.get("event", "step") for r in log]
  assert names[:5] == ["start", "layout", "state", "compile", "data"]
  # Each stage's forward runs once per microbatch; a backward fragment shared by
  # several microbatches runs once for each of them.
  programs = records.events(log, "compile")[0]["programs"]
  assert all(name.endswith(".train_step") for name in programs)
  assert max(programs.values()) > 1 and sum(programs.values()) > len(programs)
  assert names[-2:] == ["checkpoint_closed", "end"] and log[-1]["status"] == "completed"
  assert names.count("step") == 6 and names.count("eval") == 2 and names.count("save") == 2
  assert all("time" in r and "tokens_per_s_per_gpu" in r for r in records.steps(log))
  env = json.loads((run / "env.json").read_text())
  assert {"software", "configuration", "hardware", "rank_map"} <= set(env)
  assert {d["mesh"] for d in env["rank_map"]} == {"mesh0", "mesh1"}
  results = json.loads((run / "results.json").read_text())
  assert results["checkpoint"]["saves"] == 2 and len(results["eval"]) == 2
  assert results["startup_s"]["compile"] > 0
