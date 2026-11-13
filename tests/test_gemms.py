"""GEMMs from optimized HLO: library calls, Triton fusions and plain dots, their
shapes as C[M, N] row major, and launches per run through loops and branches."""

import ast
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from jax_titan.observability import gemms

REPO = pathlib.Path(__file__).resolve().parents[1]

_GEMM_CONFIG = (
    '{"operation_queue_id":"0","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,'
    '"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["%s"],"rhs_contracting_dimensions":'
    '["%s"],"lhs_batch_dimensions":[%s],"rhs_batch_dimensions":[%s]},"alpha_imag":0,"precision_config":'
    '{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT",'
    '"grad_x":false,"grad_y":false,"damax_output":false,"selected_algorithm":"7"},'
    '"force_earliest_schedule":false}')
_MATMUL = _GEMM_CONFIG % (1, 0, "", "")
_BATCHED = _GEMM_CONFIG % (2, 1, '"0"', '"0"')

# A scan over 32 layers with a hipBLASLt GEMM per layer, a rocBLAS batched GEMM
# in one branch of a conditional, a ragged dot and a GEMM with a column-major
# output, printed the way XLA dumps them.
LOOP_MODULE = f"""HloModule jit_step, is_scheduled=true, entry_computation_layout={{(bf16[8192,4096]{{1,0}})->bf16[8192,14336]{{1,0}}}}

FileNames
1 "/w/jax_titan/models/llama.py"

FunctionNames
1 "Mlp.__call__"
2 "Block.__call__"

FileLocations
1 {{file_name_id=1 function_name_id=2 line=200 end_line=200 column=4 end_column=20}}
2 {{file_name_id=1 function_name_id=1 line=120 end_line=120 column=4 end_column=20}}

StackFrames
1 {{file_location_id=1 parent_frame_id=1}}
2 {{file_location_id=2 parent_frame_id=2}}


%body (p: (s32[], bf16[8192,4096], bf16[32,4096,14336])) -> (s32[], bf16[8192,4096], bf16[32,4096,14336]) {{
  %p = (s32[], bf16[8192,4096]{{1,0}}, /*index=2*/bf16[32,4096,14336]{{2,1,0}}) parameter(0)
  %w = bf16[4096,14336]{{1,0}} fusion(%p#2, %p#0), kind=kLoop, calls=%slice_computation
  %gemm = (bf16[8192,14336]{{1,0}}, s8[33554432]{{0}}) custom-call(%p#1, %w), custom_call_target="__cublas$lt$matmul", metadata={{op_name="jit(step)/jvp()/while/body/dot_general" stack_frame_id=2}}, backend_config={_MATMUL}
  %provider = bf16[8192,14336]{{1,0}} custom-call(%p#1, %w), custom_call_target="aiter_gemm_a16w16_asm", api_version=API_VERSION_TYPED_FFI
  ROOT %t = (s32[], bf16[8192,4096]{{1,0}}, bf16[32,4096,14336]{{2,1,0}}) tuple(%p#0, %p#1, %p#2)
}}

%slice_computation (param_0: bf16[32,4096,14336], param_1: s32[]) -> bf16[4096,14336] {{
  %param_0 = bf16[32,4096,14336]{{2,1,0}} parameter(0)
  %param_1 = s32[] parameter(1)
  ROOT %slice = bf16[4096,14336]{{1,0}} dynamic-slice(%param_0, %param_1), dynamic_slice_sizes={{1,4096,14336}}
}}

%cond (c: (s32[], bf16[8192,4096], bf16[32,4096,14336])) -> pred[] {{
  %c = (s32[], bf16[8192,4096]{{1,0}}, bf16[32,4096,14336]{{2,1,0}}) parameter(0)
  ROOT %lt = pred[] constant(true)
}}

%branch_0 (a: f32[8,64,32]) -> f32[8,64,16] {{
  %a = f32[8,64,32]{{2,1,0}} parameter(0)
  %b = f32[8,32,16]{{2,1,0}} constant({{...}})
  %bmm = (f32[8,64,16]{{2,1,0}}, s8[4096]{{0}}) custom-call(%a, %b), custom_call_target="__cublas$gemm", backend_config={_BATCHED}
  ROOT %r = f32[8,64,16]{{2,1,0}} get-tuple-element(%bmm), index=0
}}

%branch_1 (a.1: f32[8,64,32]) -> f32[8,64,16] {{
  %a.1 = f32[8,64,32]{{2,1,0}} parameter(0)
  ROOT %z = f32[8,64,16]{{2,1,0}} constant(0)
}}

ENTRY %main (x: bf16[8192,4096]) -> bf16[8192,14336] {{
  %x = bf16[8192,4096]{{1,0}} parameter(0)
  %ws = bf16[32,4096,14336]{{2,1,0}} parameter(1)
  %init = (s32[], bf16[8192,4096]{{1,0}}, bf16[32,4096,14336]{{2,1,0}}) tuple(%zero, %x, %ws)
  %while = (s32[], bf16[8192,4096]{{1,0}}, bf16[32,4096,14336]{{2,1,0}}) while(%init), condition=%cond, body=%body, backend_config={{"known_trip_count":{{"n":"32"}}}}
  %pred = pred[] parameter(2)
  %a3 = f32[8,64,32]{{2,1,0}} parameter(3)
  %branch = f32[8,64,16]{{2,1,0}} conditional(%pred, %a3, %a3), branch_computations={{%branch_0, %branch_1}}
  %tokens = bf16[512,2048]{{1,0}} parameter(4)
  %experts = bf16[8,2048,1024]{{2,1,0}} parameter(5)
  %sizes = s32[8]{{0}} parameter(6)
  %moe = bf16[512,1024]{{1,0}} ragged-dot(%tokens, %experts, %sizes), lhs_contracting_dims={{1}}, rhs_contracting_dims={{1}}, lhs_ragged_dims={{0}}, rhs_group_dims={{0}}
  %l = f32[256,64]{{1,0}} parameter(7)
  %r = f32[64,128]{{1,0}} parameter(8)
  %colmajor = (f32[256,128]{{0,1}}, s8[1024]{{0}}) custom-call(%l, %r), custom_call_target="__cublas$lt$matmul", backend_config={_MATMUL}
  ROOT %out = bf16[8192,14336]{{1,0}} get-tuple-element(%while), index=1
}}
"""

# A fragment from a Shardy MPMD pipeline on gfx950: XLA swapped the operands of
# x[16,64] @ w[64,128] inside the Triton fusion and wrote the result column major.
TRITON_MODULE = """HloModule p0_stage0_fwd.step, is_scheduled=true

FileNames
1 "/w/examples/spike.py"

FunctionNames
1 "stage0"

FileLocations
1 {file_name_id=1 function_name_id=1 line=55 end_line=55 column=23 end_column=29}

StackFrames
1 {file_location_id=1 parent_frame_id=1}


%fused_computation.gemm_fusion_dot_computation (param_0: f32[16,64], param_1: f32[64,128]) -> f32[16,128] {
  %param_1 = f32[64,128]{1,0} parameter(1), metadata={scheduling_name="param_1"}
  %param_0 = f32[16,64]{1,0} parameter(0), metadata={scheduling_name="param_0"}
  %dot.2 = f32[128,16]{0,1} dot(%param_1, %param_0), lhs_contracting_dims={0}, rhs_contracting_dims={1}, metadata={op_name="jit(step)/jvp(stage0)/dot_general" scheduling_name="dot.2" stack_frame_id=1}, backend_config={"sizes":["64"]}
  ROOT %bitcast = f32[16,128]{1,0} bitcast(%dot.2), metadata={op_name="jit(step)/jvp(stage0)/dot_general" scheduling_name="bitcast" stack_frame_id=1}
}

ENTRY %main.1_spmd (param.1: f32[64,128], param: f32[16,64]) -> f32[16,128] {
  %param.1 = f32[64,128]{1,0} parameter(0)
  %param = f32[16,64]{1,0} parameter(1)
  ROOT %gemm_fusion_dot = f32[16,128]{1,0} fusion(%param, %param.1), kind=kCustom, calls=%fused_computation.gemm_fusion_dot_computation, metadata={op_name="jit(step)/jvp(stage0)/dot_general" scheduling_name="gemm_fusion_dot" stack_frame_id=1}, backend_config={"device_type":"DEVICE_TYPE_INVALID","force_earliest_schedule":false,"fusion_backend_config":{"block_level_fusion_config":{"group_size":0,"is_tma_allowed":false,"num_ctas":1,"num_stages":1,"num_warps":"4","output_tiles":[{"sizes":["16","128"]}]},"kind":"__triton_nested_gemm_fusion"},"operation_queue_id":"0"}
}
"""


def _by_kernel(text, platform="rocm"):
  return {s["kernel"]: s for s in gemms.gemms(gemms.Module(text, platform=platform))}


def test_library_gemms_through_a_loop_and_a_branch():
  module = gemms.Module(LOOP_MODULE, platform="rocm")
  sites = {s["kernel"]: s for s in gemms.gemms(module)}
  assert set(sites) == {"gemm", "bmm", "moe", "colmajor"}
  assert module.other_calls == {"aiter_gemm_a16w16_asm": 32}
  layer = sites["gemm"]
  assert (layer["library"], layer["count"], layer["flags"]) == ("hipBLASLt", 32, [])
  assert (layer["m"], layer["n"], layer["k"], layer["batch"]) == (8192, 14336, 4096, 1)
  assert (layer["a_dtype"], layer["c_dtype"], layer["layout"]) == ("bf16", "bf16", "MK,KN")
  assert layer["flops"] == 2 * 8192 * 14336 * 4096
  assert layer["source"] == "llama.py:120 Mlp.__call__" and layer["pass"] == "fwd"
  assert layer["config"] == {"algorithm": 7, "workspace_bytes": 33554432}
  assert gemms.config_text(layer) == "algorithm 7, workspace 32 MiB"

  bmm = sites["bmm"]
  assert (bmm["library"], bmm["count"], bmm["flags"]) == ("rocBLAS", 1, ["conditional"])
  assert (bmm["batch"], bmm["m"], bmm["n"], bmm["k"]) == (8, 64, 16, 32)

  moe = sites["moe"]
  assert (moe["library"], moe["groups"], moe["m"], moe["n"], moe["k"]) == ("XLA", 8, 512, 1024, 2048)
  assert moe["flops"] == 2 * 512 * 1024 * 2048


def test_column_major_output_swaps_the_operands():
  s = _by_kernel(LOOP_MODULE)["colmajor"]
  # C[256,128] column major is C^T[128,256] row major: rhs^T [128,64] times lhs^T [64,256].
  assert (s["m"], s["n"], s["k"], s["layout"], s["swapped"]) == (128, 256, 64, "KM,NK", True)


def test_triton_fusion_in_a_pipeline_fragment():
  (s,) = gemms.gemms(gemms.Module(TRITON_MODULE))
  assert (s["module"], s["kernel"], s["library"], s["op"]) == ("p0_stage0_fwd.step", "gemm_fusion_dot", "Triton", "dot")
  assert (s["m"], s["n"], s["k"], s["layout"]) == (16, 128, 64, "MK,KN")
  assert s["config"] == {"output_tile": "16x128", "block_m": 16, "block_n": 128, "block_k": 64,
                         "num_warps": 4, "num_stages": 1}
  assert gemms.config_text(s) == "16x128x64, warps 4, stages 1"
  assert s["source"] == "spike.py:55 stage0"


# MI300X and MI325X: FP8 in the fnuz formats, with the scales after the operands.
GFX942_MODULE = """HloModule jit_fp8_step, is_scheduled=true

ENTRY %main (a: f8e4m3fnuz[4096,8192], b: f8e4m3fnuz[8192,4096], sa: f32[], sb: f32[]) -> bf16[4096,4096] {
  %a = f8e4m3fnuz[4096,8192]{1,0} parameter(0)
  %b = f8e4m3fnuz[8192,4096]{1,0} parameter(1)
  %sa = f32[] parameter(2)
  %sb = f32[] parameter(3)
  %fp8 = (bf16[4096,4096]{1,0}, s8[67108864]{0}) custom-call(%a, %b, %sa, %sb), custom_call_target="__cublas$lt$matmul$f8", backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"epilogue":"DEFAULT","scale_mode":1,"selected_algorithm":"2"}}
  ROOT %r = bf16[4096,4096]{1,0} get-tuple-element(%fp8), index=0
}
"""


def test_mi300_fp8_gemm(tmp_path):
  (tmp_path / "module_0003.jit_fp8_step.gfx942_gpu_after_optimizations.txt").write_text(GFX942_MODULE)
  ((module, (s,)),) = gemms.analyze(tmp_path)
  assert (module.platform, s["library"], s["op"]) == ("rocm", "hipBLASLt", "__cublas$lt$matmul$f8")
  assert (s["a_dtype"], s["b_dtype"], s["c_dtype"]) == ("f8e4m3fnuz", "f8e4m3fnuz", "bf16")
  assert (s["m"], s["n"], s["k"]) == (4096, 4096, 8192)
  assert s["config"] == {"algorithm": 2, "workspace_bytes": 67108864, "scale_mode": 1}


def test_thunk_sequence_confirms_the_launches(tmp_path):
  name = "module_0008.p0_stage0_fwd.step"
  (tmp_path / f"{name}.gfx950_gpu_after_optimizations.txt").write_text(TRITON_MODULE)
  (tmp_path / f"{name}.thunk_sequence.txt").write_text(
      "001: kCustomKernel [source | sink] kernel=gemm_fusion_dot, profile_annotation=gemm_fusion_dot, "
      "launch dimensions=blocks: {4, 1, 1}, threads/block: {256, 1, 1}, cluster_dim=nullopt, shmem_bytes=32768\n")
  ((module, (s,)),) = gemms.analyze(tmp_path)
  assert s["launch"] == {"grid": 4, "threads": 256, "lds_bytes": 32768}
  assert gemms.config_text(s) == "16x128x64, warps 4, stages 1, grid 4, LDS 32 KiB"
  assert module.launch_check == {"file": f"{name}.thunk_sequence.txt", "kernels": 1, "kernels_launched": 1,
                                 "library_calls": 0, "library_thunks": 0}

  (tmp_path / "module_0007.jit_step.gfx950_gpu_after_optimizations.txt").write_text(LOOP_MODULE)
  (tmp_path / "module_0007.jit_step.thunk_sequence.txt").write_text(
      "001: kWhile [source | sink]\n    002: kCublasLtMatmul [source | sink] kCublasLtMatmul\n"
      "003: kGemm [source | sink] kGemm\n")
  checks = {m.name: m.launch_check for m, _ in gemms.analyze(tmp_path)}
  # Three library calls in the HLO, two library thunks: the dump is not of one executable.
  assert (checks["jit_step"]["library_calls"], checks["jit_step"]["library_thunks"]) == (3, 2)
  assert "MISMATCH" in gemms._launch_text(checks.values())


def test_group_sums_launches_per_shape():
  sites = gemms.gemms(gemms.Module(LOOP_MODULE)) + gemms.gemms(gemms.Module(TRITON_MODULE))
  rows = gemms.group(sites + [dict(sites[0], kernel="again", module="other", count=2)])
  top = rows[0]
  assert (top["library"], top["count"], top["kernels"], top["modules"]) == ("hipBLASLt", 34, 2, ["jit_step", "other"])
  assert top["total_flops"] == 34 * top["flops"]
  assert [r["total_flops"] for r in rows] == sorted((r["total_flops"] for r in rows), reverse=True)


def test_reads_only_the_final_program(tmp_path):
  (tmp_path / "module_0007.jit_step.gfx950_gpu_after_optimizations.txt").write_text(LOOP_MODULE)
  (tmp_path / "module_0008.p0_stage0_fwd.step.gfx950_gpu_after_optimizations.txt").write_text(TRITON_MODULE)
  for name in ("module_0007.jit_step.before_optimizations.txt",
               "module_0007.jit_step.before_config_assignment.txt",
               "module_0007.jit_step.gfx950_gpu_after_optimizations-buffer-assignment.txt",
               "module_0007.jit_step.autotune_results.pbtxt"):
    (tmp_path / name).write_text(TRITON_MODULE.replace("p0_stage0_fwd.step", "wrong"))
  results = gemms.analyze(tmp_path)
  assert [(m.name, m.platform, len(s)) for m, s in results] == [("jit_step", "rocm", 4), ("p0_stage0_fwd.step", "rocm", 1)]
  assert [m.name for m, _ in gemms.analyze(tmp_path, "stage0")] == ["p0_stage0_fwd.step"]

  out_csv, out_json = tmp_path / "gemms.csv", tmp_path / "gemms.json"
  assert gemms.main([str(tmp_path), "--csv", str(out_csv), "--json", str(out_json)]) == 0
  rows = list(csv.DictReader(out_csv.open()))
  assert len(rows) == 5 and {r["library"] for r in rows} == {"hipBLASLt", "rocBLAS", "XLA", "Triton"}
  report = json.loads(out_json.read_text())
  assert [m["module"] for m in report["modules"]] == ["jit_step", "p0_stage0_fwd.step"]
  assert report["modules"][0]["shapes"][0]["count"] == 32
  assert gemms.main([str(tmp_path / "missing")]) == 1


def test_launches_per_training_step(tmp_path):
  dump, run = tmp_path / "dump", tmp_path / "run"
  dump.mkdir()
  run.mkdir()
  (dump / "module_0007.jit_step.gfx950_gpu_after_optimizations.txt").write_text(LOOP_MODULE)
  (dump / "module_0008.p0_stage0_fwd.step.gfx950_gpu_after_optimizations.txt").write_text(TRITON_MODULE)
  (dump / "module_0009.jit_init.gfx950_gpu_after_optimizations.txt").write_text(
      TRITON_MODULE.replace("p0_stage0_fwd.step", "jit_init"))
  events = [{"event": "start", "time": 0}, {"event": "compile", "time": 1, "lower_s": 1, "compile_s": 2,
                                            "programs": {"p0_stage0_fwd.step": 8, "p1_stage1_bwd.step": 8}}]
  (run / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
  out = tmp_path / "step.json"
  assert gemms.main([str(dump), "--run_dir", str(run), "--merge", "--json", str(out)]) == 0
  report = json.loads(out.read_text())
  assert report["counts"] == "per step"
  (module,) = report["modules"]
  assert (module["module"], module["calls_per_step"]) == ("p0_stage0_fwd.step", 8)
  assert module["gemms"][0]["count"] == 8 and module["shapes"][0]["total_flops"] == 8 * 2 * 16 * 128 * 64


def test_calls_per_step_from_a_program_or_json(tmp_path):
  ifrt = ("%0 = ifrt.Call @p0_stage0_fwd.step(%a) on devices [0]\n"
          "%1 = ifrt.Call @p1_stage1_bwd.step(%b) on devices [1]\n"
          "%2 = ifrt.Call @p1_stage1_bwd.step(%c) on devices [1]\n")
  assert gemms.program_calls(ifrt) == {"p0_stage0_fwd.step": 1, "p1_stage1_bwd.step": 2}
  mpmd = '%0 = mpmd.fragment_call<mesh="m0", origin=["stage0"]> @p0_stage0_fwd.step(%a)\n'
  assert gemms.program_calls(mpmd) == {"p0_stage0_fwd.step": 1}
  assert gemms.program_calls("module @jit_train_step attributes {} {\n}") == {"jit_train_step": 1}
  (tmp_path / "calls.json").write_text('{"jit_step": 4}')
  assert gemms.programs_from_file(tmp_path / "calls.json") == {"jit_step": 4}


def test_runs_without_jax_or_jax_titan(tmp_path):
  source = pathlib.Path(gemms.__file__)
  # Only the standard library, so the one file runs anywhere with Python 3.8+.
  tree = ast.parse(source.read_text(), feature_version=(3, 8))
  imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
  imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
  assert imported <= sys.stdlib_module_names | {"__future__"}, imported - sys.stdlib_module_names

  dump = tmp_path / "dump"
  dump.mkdir()
  (dump / "module_0008.p0_stage0_fwd.step.gfx942_gpu_after_optimizations.txt").write_text(TRITON_MODULE)
  copy = shutil.copy(source, tmp_path / "hlo_gemms.py")
  out = tmp_path / "out.json"
  # -I -S: no site-packages, so no JAX and no JAX-Titan to import.
  result = subprocess.run([sys.executable, "-I", "-S", copy, str(dump), "--sites", "--json", str(out)],
                          capture_output=True, text=True, check=False, cwd=tmp_path)
  assert result.returncode == 0, result.stderr
  assert "gemm_fusion_dot" in result.stdout and "16x128x64, warps 4, stages 1" in result.stdout
  assert json.loads(out.read_text())["modules"][0]["platform"] == "rocm"


@pytest.mark.parametrize("parallel", ["fsdp", "mpmd"])
def test_a_trainer_run_spmd_or_mpmd(tmp_path, parallel):
  dump, run = tmp_path / "dump", tmp_path / "run"
  env = dict(os.environ, JAX_ENABLE_COMPILATION_CACHE="false",
             XLA_FLAGS=f"--xla_dump_to={dump} --xla_dump_hlo_as_text")
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", "--platform=cpu", "--model=llama3-test",
       f"--parallel={parallel}", "--data=synthetic", "--seq_len=32", "--microbatches=4",
       "--microbatch_size=4", "--steps=1", "--warmup_steps=1", f"--run_dir={run}", "--power_sample_s=0"],
      capture_output=True, text=True, check=False, cwd=REPO, env=env)
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  programs = gemms.programs_from_run(run)
  modules = gemms.per_step(gemms.analyze(dump), programs)
  # Every program the step runs is in the dump, whatever the parallelism.
  assert sorted(m.name for m, _ in modules) == sorted(programs)
  # The LM head's forward GEMM on one device, tokens x vocabulary: SPMD runs the
  # step's 16 sequences of 32 tokens as one batch over 8 devices, and the
  # pipeline runs 4 microbatches, each split over a stage's 4 devices.
  head = [s for _, ss in modules for s in ss if (s["n"], s["k"]) == (128256, 256)]
  tokens, launches = (64, 1) if parallel == "fsdp" else (32, 4)
  assert {s["m"] for s in head} == {tokens} and sum(s["count"] for s in head) == launches
  if parallel == "fsdp":
    assert programs == {"jit_train_step": 1}
  else:
    assert max(programs.values()) > 1


def test_compiled_scan_counts_each_layer():
  def loss(ws, x):
    def layer(h, w):
      return jnp.tanh(h @ w), None
    h, _ = jax.lax.scan(layer, x, ws)
    return h.sum()

  ws, x = jnp.ones((3, 256, 256), jnp.float32), jnp.ones((64, 256), jnp.float32)
  text = jax.jit(jax.grad(loss)).lower(ws, x).compile().as_text()
  sites = gemms.gemms(gemms.Module(text, platform="cpu"))
  shapes = sorted((s["m"], s["n"], s["k"], s["layout"], s["pass"], s["count"]) for s in sites)
  # Forward h @ w, and backward dy @ w^T and h^T @ dy, once per layer.
  assert shapes == [(64, 256, 256, "MK,KN", "fwd", 3), (64, 256, 256, "MK,NK", "bwd", 3),
                    (256, 256, 64, "KM,KN", "bwd", 3)]
  assert all(s["source"].startswith("test_gemms.py:") and s["source"].endswith(".layer") for s in sites)
