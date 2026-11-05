"""Pretrains a Llama model with Shardy MPMD pipeline parallelism.

  python -m jaxtitan.trainer.train --model=llama3.1-8b --num_stages=2

`--parallel=dp` and `--parallel=fsdp` train the same model on the same batches
without the pipeline, as one `jax.jit` program over all devices.

With `--checkpoint_dir`, training resumes from the latest checkpoint there.
"""

import argparse
import dataclasses
import functools
import json
import os
import time
from typing import Any, Callable

from jaxtitan import paths

_NORM_PARAMS = ("attn_norm", "mlp_norm", "final_norm")


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="llama3.1-8b")
  parser.add_argument(
      "--parallel", default="mpmd", choices=["mpmd", "dp", "fsdp"],
      help="mpmd: Shardy MPMD pipeline, data parallel inside each stage. "
      "dp, fsdp: no pipeline; one jax.jit program over all devices with "
      "replicated or sharded parameters and optimizer state.",
  )
  parser.add_argument(
      "--num_stages", type=int, default=2,
      help="Pipeline stages. dp and fsdp start from the same weights as the "
      "pipeline with this many stages.",
  )
  parser.add_argument(
      "--schedule", default="ONE_FWD_ONE_BWD",
      help="Shardy pipeline schedule name, or empty for the default order.",
  )
  parser.add_argument(
      "--stages_per_mesh", type=int, default=1,
      help="Pipeline stages on each mesh; above 1 for circular schedules, "
      "which put stage i on mesh i %% (num_stages / stages_per_mesh).",
  )
  parser.add_argument(
      "--mpmd_options", default="",
      help="JSON object of extra Shardy MPMD partitioning options, e.g. "
      '\'{"mpmd_merge_inferred_after_scheduling": true}\'.',
  )
  parser.add_argument(
      "--reduce_grads", default="step", choices=["microbatch", "step"],
      help="When mpmd stages sum gradients across their data-parallel ranks: "
      "in every microbatch's backward pass, or once per step from per-rank "
      "parameter copies.",
  )
  parser.add_argument(
      "--stage_fsdp", action="store_true",
      help="mpmd: shard parameters and optimizer state across each stage's "
      "data-parallel ranks.",
  )
  parser.add_argument(
      "--tensor_parallel", type=int, default=1,
      help="Tensor-parallel degree, inside each stage for mpmd.",
  )
  parser.add_argument("--microbatches", type=int, default=8)
  parser.add_argument(
      "--microbatch_size", type=int, default=4,
      help="Sequences per microbatch, sharded across a stage's data-parallel "
      "ranks. dp and fsdp run all microbatches as one batch.",
  )
  parser.add_argument("--seq_len", type=int, default=2048)
  parser.add_argument("--steps", type=int, default=50)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--warmup_steps", type=int, default=10)
  parser.add_argument(
      "--data",
      default=str(paths.work_dir() / "data/wikitext103_train.llama3.bin"),
      help="uint32 token file, or 'synthetic'.",
  )
  parser.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      "--peak_tflops_per_device", type=float, default=2500.0,
      help="Dense BF16 peak used for MFU; 2500 for MI355X.",
  )
  parser.add_argument("--log_file", default="")
  parser.add_argument(
      "--checkpoint_dir", default="",
      help="Saves training state here, and resumes from its latest checkpoint.",
  )
  parser.add_argument(
      "--checkpoint_every", type=int, default=0,
      help="Steps between checkpoints; 0 saves only after the last step.",
  )
  parser.add_argument(
      "--profile_steps", default="",
      help="START:END traces steps [START, END) with jax.profiler into "
      "--profile_dir.",
  )
  parser.add_argument(
      "--profile_dir", default=str(paths.work_dir() / "profiles"),
  )
  parser.add_argument("--platform", default="", help="'cpu' runs on 8 CPU devices.")
  parser.add_argument("--seed", type=int, default=0)
  return parser.parse_args(argv)


@dataclasses.dataclass
class Layout:
  """How the training state is placed and stepped.

  `abstract_state` has the tree structure, shapes, dtypes and shardings of
  `init()`'s `(params, opt_state)`, so a checkpoint can be restored without
  initializing first.
  """

  init: Callable[[], tuple]
  abstract_state: tuple
  step: Any
  batch_spec: Any
  put_batch: Callable
  structure: str


def _decay_mask(params):
  import jax  # pylint: disable=g-import-not-at-top

  return jax.tree_util.tree_map_with_path(
      lambda path, _: not any(getattr(k, "key", None) in _NORM_PARAMS for k in path),
      params,
  )


def _abstract(shapes, shardings):
  import jax  # pylint: disable=g-import-not-at-top

  return jax.tree.map(
      lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), shapes, shardings
  )


def _build_mpmd(args, cfg, devices, optimizer) -> Layout:
  """Shardy MPMD pipeline: one mesh per stage, DP, FSDP and TP inside each."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan import parallel  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  num_stages = args.num_stages
  num_meshes = num_stages // args.stages_per_mesh
  topology = jt_mpmd.make_topology(
      num_meshes, devices, tensor_parallel=args.tensor_parallel
  )
  options = json.loads(args.mpmd_options) if args.mpmd_options else None
  config = jt_mpmd.make_config(
      topology, args.schedule or None, options, num_stages=num_stages
  )
  meshes = [topology[jt_mpmd.mesh_name(i % num_meshes)] for i in range(num_stages)]
  data_parallel = meshes[0].shape[jt_mpmd.DATA_AXIS]
  if args.microbatch_size % data_parallel:
    raise ValueError(
        f"--microbatch_size={args.microbatch_size} does not split across "
        f"{data_parallel} data-parallel ranks per stage"
    )
  activation_sharding = NamedSharding(meshes[0], P(jt_mpmd.DATA_AXIS))

  def constrain(h):
    return jax.lax.with_sharding_constraint(h, activation_sharding)

  per_step = args.reduce_grads == "step"
  stage_fns = []
  for i in range(num_stages):
    fn = functools.partial(
        llama.stage_forward, cfg=cfg, stage=i, num_stages=num_stages,
        remat=args.remat, constrain=None if per_step else constrain,
    )
    if i < num_stages - 1:
      stage_fns.append(functools.partial(lambda f, p, x: f(p, x, None), fn))
    else:
      stage_fns.append(fn)

  def init_stage(i, key):
    params = llama.init_stage_params(cfg, key, stage=i, num_stages=num_stages)
    return params, optimizer.init(params)

  keys = jax.random.split(jax.random.key(args.seed), num_stages)
  shapes = [
      jax.eval_shape(functools.partial(init_stage, i), keys[i])
      for i in range(num_stages)
  ]
  specs = [
      parallel.tree_specs(
          shapes[i], llama.logical_axes, meshes[i].shape, fsdp=args.stage_fsdp
      )
      for i in range(num_stages)
  ]
  shardings = [
      jax.tree.map(lambda s, m=meshes[i]: NamedSharding(m, s), specs[i])
      for i in range(num_stages)
  ]

  def init():
    states = [
        jax.jit(functools.partial(init_stage, i), out_shardings=shardings[i])(keys[i])
        for i in range(num_stages)
    ]
    return tuple(s[0] for s in states), tuple(s[1] for s in states)

  param_shardings = tuple(s[0] for s in shardings)
  opt_shardings = tuple(s[1] for s in shardings)
  abstract_state = (
      tuple(_abstract(shapes[i][0], param_shardings[i]) for i in range(num_stages)),
      tuple(_abstract(shapes[i][1], opt_shardings[i]) for i in range(num_stages)),
  )

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  token_sharding = NamedSharding(meshes[0], P(None, jt_mpmd.DATA_AXIS, None))
  target_sharding = NamedSharding(meshes[-1], P(None, jt_mpmd.DATA_AXIS, None))
  loss_sharding = NamedSharding(meshes[-1], P())

  train_step = jt_mpmd.make_train_step(
      stage_fns, optimizer, mesh=meshes[0],
      copy_axes=[llama.device_copy_axes(s[0]) for s in shapes] if per_step else None,
      param_specs=[s[0] for s in specs],
  )
  jitted = mpmd.jit(
      train_step,
      config,
      in_shardings=(param_shardings, opt_shardings, token_sharding, target_sharding),
      out_shardings=(param_shardings, opt_shardings, loss_sharding),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    batch = batch.reshape(num_mb, mb, seq + 1)
    return (
        jax.device_put(batch[..., :-1], token_sharding),
        jax.device_put(batch[..., 1:], target_sharding),
    )

  batch_spec = jax.ShapeDtypeStruct((num_mb, mb, seq), jnp.int32)
  return Layout(
      init, abstract_state, jitted, batch_spec, put_batch, f"mpmd:{num_stages}"
  )


def _build_spmd(args, cfg, devices, optimizer) -> Layout:
  """No pipeline: one `jax.jit` program over all devices."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top
  from jaxtitan import parallel  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  tp = args.tensor_parallel
  mesh = jax.sharding.Mesh(
      np.asarray(devices).reshape(len(devices) // tp, tp),
      (parallel.DATA_AXIS, parallel.TENSOR_AXIS),
  )
  batch_sharding = NamedSharding(mesh, P(parallel.DATA_AXIS))

  def init_params(keys):
    return llama.merge_stage_params([
        llama.init_stage_params(cfg, keys[i], stage=i, num_stages=args.num_stages)
        for i in range(args.num_stages)
    ])

  def init_state(keys):
    params = init_params(keys)
    return params, optimizer.init(params)

  keys = jax.random.split(jax.random.key(args.seed), args.num_stages)
  shapes = jax.eval_shape(init_state, keys)
  shardings = jax.tree.map(
      lambda s: NamedSharding(mesh, s),
      parallel.tree_specs(
          shapes, llama.logical_axes, mesh.shape, fsdp=args.parallel == "fsdp"
      ),
  )

  def init():
    return jax.jit(init_state, out_shardings=shardings)(keys)

  loss_fn = functools.partial(
      llama.stage_forward, cfg=cfg, stage=0, num_stages=1, remat=args.remat,
      constrain=lambda h: jax.lax.with_sharding_constraint(h, batch_sharding),
  )

  def train_step(params, opt_state, tokens, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

  param_shardings, opt_shardings = shardings
  jitted = jax.jit(
      train_step,
      in_shardings=(param_shardings, opt_shardings, batch_sharding, batch_sharding),
      out_shardings=(param_shardings, opt_shardings, NamedSharding(mesh, P())),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    return (
        jax.device_put(batch[:, :-1], batch_sharding),
        jax.device_put(batch[:, 1:], batch_sharding),
    )

  batch = args.microbatches * args.microbatch_size
  batch_spec = jax.ShapeDtypeStruct((batch, args.seq_len), jnp.int32)
  return Layout(
      init, _abstract(shapes, shardings), jitted, batch_spec, put_batch, "spmd"
  )


def main(argv=None):
  args = parse_args(argv)
  if args.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top

  if args.platform == "cpu":
    jax.config.update("jax_num_cpu_devices", 8)

  import jaxtitan  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import
  from jaxtitan.data.text import WindowSampler  # pylint: disable=g-import-not-at-top
  from jaxtitan.models import llama  # pylint: disable=g-import-not-at-top

  cfg = llama.FLAVORS[args.model]
  devices = jax.devices()
  tp = f", tensor parallel {args.tensor_parallel}" if args.tensor_parallel > 1 else ""
  if args.parallel == "mpmd":
    layout = (
        f"{args.num_stages} stages{' with FSDP' if args.stage_fsdp else ''}{tp} | "
        f"schedule={args.schedule or 'default'} | "
        f"gradients reduced per {args.reduce_grads}"
    )
    if args.mpmd_options:
      layout += f" | {args.mpmd_options}"
  else:
    layout = f"{args.parallel} over {len(devices)} devices{tp}, no pipeline"
  print(
      f"jax {jax.__version__} | {len(devices)} x {devices[0].device_kind} | "
      f"{args.model} ({cfg.num_params() / 1e9:.2f}B params) | {layout}",
      flush=True,
  )

  lr = optax.warmup_cosine_decay_schedule(
      0.0, args.lr, args.warmup_steps, max(args.steps, args.warmup_steps + 1),
      end_value=args.lr * 0.1,
  )
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1, mask=_decay_mask),
  )

  build = _build_mpmd if args.parallel == "mpmd" else _build_spmd
  layout = build(args, cfg, devices, optimizer)

  ckpt = None
  start = 0
  if args.checkpoint_dir:
    from jaxtitan.trainer import checkpoint  # pylint: disable=g-import-not-at-top

    ckpt = checkpoint.Checkpointer(args.checkpoint_dir, save_every=args.checkpoint_every)
    start = ckpt.latest_step() or 0
  t0 = time.time()
  if start:
    (params, opt_states), meta = ckpt.restore(start, layout.abstract_state)
    if meta["structure"] != layout.structure or meta["model"] != args.model:
      raise ValueError(
          f"checkpoint holds {meta['model']} as {meta['structure']}, not "
          f"{args.model} as {layout.structure}"
      )
    print(
        f"restored step {start} from {args.checkpoint_dir} in {time.time() - t0:.1f}s",
        flush=True,
    )
  else:
    params, opt_states = layout.init()
    print(f"initialized params and optimizer state in {time.time() - t0:.1f}s", flush=True)

  t0 = time.time()
  lowered = layout.step.lower(params, opt_states, layout.batch_spec, layout.batch_spec)
  t1 = time.time()
  compiled = lowered.compile()
  print(
      f"lowered in {t1 - t0:.1f}s, compiled in {time.time() - t1:.1f}s",
      flush=True,
  )

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  if args.data == "synthetic":

    def next_batch(step):
      rng = np.random.default_rng((args.seed, step))
      return rng.integers(0, cfg.vocab_size, (num_mb * mb, seq + 1), dtype=np.int32)

  else:
    sampler = WindowSampler(args.data, seq, seed=args.seed, step=start)
    print(f"data: {args.data} ({sampler.num_tokens / 1e6:.1f}M tokens)", flush=True)

    def next_batch(step):
      assert sampler.step == step, (sampler.step, step)
      return sampler.next(num_mb * mb)

  def meta(step):
    return {
        "step": step, "model": args.model, "structure": layout.structure,
        "data": {"seed": args.seed, "step": step},
    }

  profile = [int(s) for s in args.profile_steps.split(":")] if args.profile_steps else None
  tokens_per_step = num_mb * mb * seq
  flops_per_step = cfg.flops_per_token(seq) * tokens_per_step
  peak_flops = len(devices) * args.peak_tflops_per_device * 1e12
  log = open(args.log_file, "a") if args.log_file else None

  for step in range(start, args.steps):
    if profile and step == profile[0]:
      jax.block_until_ready((params, opt_states))
      jax.profiler.start_trace(args.profile_dir)
    tokens, targets = layout.put_batch(next_batch(step))
    t0 = time.time()
    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      params, opt_states, loss = compiled(params, opt_states, tokens, targets)
      loss = float(loss)
    dt = time.time() - t0
    if profile and step == profile[1] - 1:
      jax.block_until_ready((params, opt_states))
      jax.profiler.stop_trace()
      print(f"traced steps {profile[0]}-{step} into {args.profile_dir}", flush=True)
    record = {
        "step": step,
        "loss": loss,
        "step_time_s": dt,
        "tokens_per_s": tokens_per_step / dt,
        "mfu": flops_per_step / dt / peak_flops,
    }
    print(
        f"step {step:4d} | loss {loss:8.4f} | {dt * 1e3:8.1f} ms | "
        f"{record['tokens_per_s']:10.0f} tok/s | MFU {100 * record['mfu']:5.1f}%",
        flush=True,
    )
    if log:
      log.write(json.dumps(record) + "\n")
      log.flush()
    if step == start:
      peaks = [
          (d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30
          for d in devices
      ]
      print("peak HBM per device (GiB): " + " ".join(f"{p:.1f}" for p in peaks), flush=True)
    if not np.isfinite(loss):
      raise FloatingPointError(f"non-finite loss at step {step}")
    if ckpt:
      t0 = time.time()
      force = step + 1 == args.steps
      if ckpt.save(step + 1, (params, opt_states), meta(step + 1), force=force):
        print(f"saved step {step + 1} in {time.time() - t0:.1f}s", flush=True)
  if ckpt:
    ckpt.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
