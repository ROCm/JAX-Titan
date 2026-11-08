"""The training budget planner against published and measured numbers."""

import dataclasses
import json
import math
import pathlib

import pytest

from jaxtitan.models import llama
from jaxtitan.observability import flops as flops_lib
from jaxtitan.planning import budget
from jaxtitan.planning import comms
from jaxtitan.planning import gpus
from jaxtitan.planning import memory
from jaxtitan.planning import models

RESULTS = pathlib.Path(__file__).parent.parent / "docs" / "results" / "2026-11-07"

LLAMA3_70B = {"vocab_size": 128256, "hidden_size": 8192, "num_hidden_layers": 80,
              "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 28672}
LLAMA3_405B = {"vocab_size": 128256, "hidden_size": 16384, "num_hidden_layers": 126,
               "num_attention_heads": 128, "num_key_value_heads": 8,
               "intermediate_size": 53248}
MIXTRAL_8X7B = {"vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
                "num_attention_heads": 32, "num_key_value_heads": 8,
                "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2}


def _hf(tmp_path, name, config) -> models.Model:
  path = tmp_path / name / "config.json"
  path.parent.mkdir()
  path.write_text(json.dumps(config))
  return models.from_name(f"hf:{path}")


def test_llama_flavor_matches_the_trainers_flops():
  cfg = llama.FLAVORS["llama3.1-8b"]
  model = models.from_name("llama3.1-8b")
  assert model.params() == cfg.num_params()
  # The trainer's 6N excludes the norms' parameters, a 3e-5 difference.
  assert model.training_flops_per_token(2048) == pytest.approx(cfg.flops_per_token(2048),
                                                               rel=1e-4)
  assert model.recompute_flops_per_token(2048) == pytest.approx(
      flops_lib.recompute_flops_per_token(cfg, 2048, remat=True))


def test_published_parameter_counts(tmp_path):
  mixtral = _hf(tmp_path, "mixtral", MIXTRAL_8X7B)
  assert mixtral.params() / 1e9 == pytest.approx(46.7, rel=0.005)
  assert mixtral.params(active=True) / 1e9 == pytest.approx(12.9, rel=0.01)
  assert _hf(tmp_path, "70b", LLAMA3_70B).params() / 1e9 == pytest.approx(70.6, rel=0.005)
  assert _hf(tmp_path, "405b", LLAMA3_405B).params() / 1e9 == pytest.approx(405.9, rel=0.005)


def test_llama3_405b_compute_matches_the_paper(tmp_path):
  model = _hf(tmp_path, "405b", LLAMA3_405B)
  r = budget.estimate(model, gpus.get("H100"), 16384, tokens=15.6e12, mfu=0.4, ettr=1.0,
                      attention="none")
  assert r["model_flops"] == pytest.approx(3.8e25, rel=0.03)


def test_llama31_405b_gpu_hours(tmp_path):
  # The model card's 30.84M H100-hours lie between Llama 3's 38-43% MFU (Table 4) at an
  # ETTR of 0.9 and 1.
  model = _hf(tmp_path, "405b", LLAMA3_405B)

  def gpu_hours(mfu, ettr):
    return budget.estimate(model, gpus.get("H100"), 16384, tokens=15.6e12, mfu=mfu, ettr=ettr,
                           attention="none")["gpu_hours"]

  assert gpu_hours(0.43, 1.0) < 30.84e6 < gpu_hours(0.38, 0.9)


def test_palm_mfu_counts_attention_in_full():
  # PaLM 540B (arXiv:2204.02311, Table 3 and Appendix B): 238.3k tokens/s on 6,144 TPU v4
  # chips of 275 TFLOP/s is 46.2% MFU with 6N + 12LHQT.
  palm = models.Model("palm-540b", 256000, 18432, 118, 48, 1, 256, 73728, tied_embeddings=True)
  assert palm.params() / 1e9 == pytest.approx(540.35, rel=0.002)
  mfu = 238.3e3 * palm.training_flops_per_token(2048, "full") / (6144 * 275e12)
  assert mfu == pytest.approx(0.462, abs=0.001)
  assert palm.training_flops_per_token(2048, "causal") < palm.training_flops_per_token(
      2048, "full")


def test_one_mi350x_and_the_scaling_book_examples(tmp_path):
  model = _hf(tmp_path, "70b", LLAMA3_70B)
  one = budget.estimate(model, gpus.get("MI350X"), 1, tokens=20e12, mfu=0.4, ettr=1.0,
                        attention="none")
  assert one["wall_days"] / 365.25 == pytest.approx(286, rel=0.02)
  layout = memory.Layout(seq_len=8192)
  assert memory.per_gpu(model, layout)["state_bytes"] > 3 * 288e9
  pods = budget.estimate(model, gpus.get("TPUv5p"), 4 * 8960, tokens=15e12, mfu=0.4,
                         ettr=1.0, attention="none")
  assert pods["wall_days"] == pytest.approx(11.0, rel=0.03)


def test_tokens_and_time_budgets_invert():
  model = models.from_name("llama3.1-8b")
  forward = budget.estimate(model, gpus.get("MI325X"), 256, tokens=2e12, mfu=0.35, ettr=0.9)
  back = budget.estimate(model, gpus.get("MI325X"), 256, seconds=forward["wall_days"] * 86400,
                         mfu=0.35, ettr=0.9)
  assert back["tokens"] == pytest.approx(2e12)
  assert forward["compute_days"] == pytest.approx(0.9 * forward["wall_days"])


def test_gpu_table():
  for gpu in gpus.GPUS.values():
    assert gpu.source.startswith("https://")
    assert gpu.bf16_tflops and gpu.hbm_gb
  mi325 = gpus.get("MI325")
  assert (mi325.bf16_tflops, mi325.fp8_tflops, mi325.hbm_gb, mi325.board_power_w) == (
      1307.4, 2614.9, 256, 1000)
  assert gpus.get("AMD Instinct MI355X") is gpus.GPUS["MI355X"]
  assert gpus.get("MI450") is gpus.GPUS["MI455X"]
  assert gpus.GPUS["MI455X"].bf16_tflops * 2 == pytest.approx(10.1e3, rel=0.005)
  with pytest.raises(ValueError):
    gpus.get("MI300X").peak_tflops("fp4")


def test_memory_layers_and_ends():
  model = models.from_name("llama3.1-8b")
  one = memory.per_gpu(model, memory.Layout(pipeline_stages=1))
  assert one["params_per_gpu"] == model.params()
  two = memory.per_gpu(model, memory.Layout(pipeline_stages=2, microbatches=2))
  assert two["params_per_gpu"] == pytest.approx(
      (model.params() - model.embedding_params()) / 2 + model.vocab * model.emb)
  moe = models.Model("half-moe", 1000, 64, 4, 4, 4, 16, 256, experts=8, experts_per_token=2,
                     expert_mlp=32, moe_layers=2)
  dense = dataclasses.replace(moe, experts=0, experts_per_token=0, expert_mlp=0, moe_layers=0)
  all_moe = dataclasses.replace(moe, moe_layers=4)
  assert memory.activation_elements_per_token_layer(moe) == pytest.approx(
      (memory.activation_elements_per_token_layer(dense)
       + memory.activation_elements_per_token_layer(all_moe)) / 2)


def test_memory_floor_against_a_recorded_run():
  c = memory.calibrate(RESULTS / "llama3.1-8b_pp2_run-dir")
  assert c["layout"]["pipeline_stages"] == 2 and c["layout"]["grad_copies"]
  assert 1.8 < c["overhead"] < 2.2


def test_data_parallel_communication():
  c = comms.data_parallel_seconds(8e9, 16, 8, 504, 50)
  assert c["domains"] == 2
  assert c["inside_s"] == pytest.approx(2 * 7 / 8 * 32e9 / 504e9)
  assert c["across_s"] == pytest.approx(2 * 32e9 / 8 / 2 / 50e9)
  fsdp = comms.data_parallel_seconds(8e9, 8, 8, 504, None, fsdp=True)
  assert fsdp["bytes_per_gpu"] == pytest.approx(8e9 * (2 + 2 + 4)) and fsdp["across_s"] == 0
  assert comms.ranks_per_domain(1024, 128, 8, 8) == 1
  assert comms.ranks_per_domain(8, 4, 1, 8) == 4
  assert comms.ranks_per_domain(4608, 576, 8, 72) == 9
  # Four GPUs of a fully connected MI355X platform share three of each GPU's seven links.
  assert gpus.get("MI355X").scaleup_gbs_for(4) == pytest.approx(3 * 72)
  assert gpus.get("MI455X").scaleup_gbs_for(9) == gpus.get("MI455X").scaleup_gbs


def test_twenty_tokens_per_parameter_is_compute_optimal():
  model = models.from_name("llama3.1-8b")
  r = budget.estimate(model, gpus.get("MI300X"), 64, tokens=20 * model.params(), mfu=0.3)
  assert r["compute_optimal"]["params"] == pytest.approx(model.params())


def test_cli_compares_gpus(tmp_path, capsys):
  out = tmp_path / "plan.json"
  assert budget.main(["--model=llama3.1-8b", "--tokens=1e12", "--gpu=MI300X,MI325,MI355X",
                      "--gpus=512", f"--json={out}"]) == 0
  rows = json.loads(out.read_text())
  assert [r["gpu"] for r in rows] == ["AMD Instinct MI300X", "AMD Instinct MI325X",
                                      "AMD Instinct MI355X"]
  assert rows[0]["wall_days"] == pytest.approx(rows[1]["wall_days"])
  assert rows[0]["wall_days"] / rows[2]["wall_days"] == pytest.approx(2516.6 / 1307.4, rel=1e-3)
  assert rows[1]["energy_mwh"] > rows[0]["energy_mwh"]
  assert "AMD Instinct MI325X" in capsys.readouterr().out


def test_from_run_carries_measured_throughput(tmp_path):
  run = budget.from_run(str(RESULTS / "llama3.1-8b_pp2_run-dir"))
  assert run["device_kind"] == "AMD Instinct MI355X" and run["memory_overhead"] > 1.8
  out = tmp_path / "plan.json"
  budget.main(["--model=llama3.1-8b", "--tokens=1e11", "--gpu=MI355X,MI325X", "--gpus=8",
               "--seq_len=2048", f"--from_run={RESULTS / 'llama3.1-8b_pp2_run-dir'}",
               f"--json={out}"])
  same, other = json.loads(out.read_text())
  assert same["mfu"] * same["peak_tflops"] == pytest.approx(run["achieved_tflops_per_gpu"])
  assert other["mfu"] == pytest.approx(run["mfu"])
  assert other["mfu_source"].startswith("carried over")
  assert math.isclose(same["memory"]["overhead"], run["memory_overhead"])
