"""Trainer layouts for models from MaxText and MaxDiffusion.

`--model=maxtext:<name>` trains a MaxText model config (e.g.
`maxtext:llama3.1-8b`) on the token data, and `--model=maxdiffusion:<flavor>`
trains FLUX.1 (`flux-dev` or `flux-schnell`) on synthetic rectified-flow
batches until an image pipeline exists. Both run as a Shardy MPMD pipeline
with data parallelism inside each stage, reducing gradients in every
microbatch's backward pass, and attention in native JAX unless the ops policy
names attention. `--parallel=dp` runs the same stages, weights and batches as
one data-parallel `jax.jit` program, the no-pipeline reference.
"""

import functools
import json

from jaxtitan.trainer.layout import Layout, Workload, abstract, placement

PROVIDERS = ("maxtext", "maxdiffusion")


def _overrides(args) -> dict:
  return json.loads(args.model_overrides) if args.model_overrides else {}


def _stage_layers(args):
  return tuple(int(n) for n in args.stage_layers.split(",")) if args.stage_layers else None


def _num_params(shapes) -> int:
  import jax  # pylint: disable=g-import-not-at-top

  return sum(x.size for x in jax.tree.leaves(shapes))


def _topology(args, devices, make_mesh) -> dict:
  """One mesh per group of stages from `make_mesh(devices)`."""
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top

  num_meshes = args.num_stages // args.stages_per_mesh
  if len(devices) % num_meshes:
    raise ValueError(f"{len(devices)} devices do not split into {num_meshes} meshes")
  per_mesh = len(devices) // num_meshes
  if args.microbatch_size % per_mesh:
    raise ValueError(
        f"--microbatch_size={args.microbatch_size} does not split across "
        f"{per_mesh} data-parallel ranks per stage"
    )
  return {
      jt_mpmd.mesh_name(i): make_mesh(devices[i * per_mesh:(i + 1) * per_mesh])
      for i in range(num_meshes)
  }


def _is_pair(v) -> bool:
  import jax  # pylint: disable=g-import-not-at-top

  return isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], jax.sharding.PartitionSpec)


def _shardings(specs, mesh):
  import jax  # pylint: disable=g-import-not-at-top

  return jax.tree.map(lambda v: jax.sharding.NamedSharding(mesh, v[1]), specs, is_leaf=_is_pair)


def _shapes(specs):
  import jax  # pylint: disable=g-import-not-at-top

  return jax.tree.map(lambda v: v[0], specs, is_leaf=_is_pair)


def _init_replicated(args, model, optimizer, meshes):
  """Init and abstract state for stage state replicated on its mesh."""
  import jax  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  def replicated(tree, mesh):
    return jax.tree.map(lambda _: jax.sharding.NamedSharding(mesh, P()), tree)

  param_shapes = model.abstract(args.seed)
  opt_shapes = tuple(jax.eval_shape(optimizer.init, p) for p in param_shapes)
  param_sh = tuple(replicated(p, m) for p, m in zip(param_shapes, meshes))
  opt_sh = tuple(replicated(s, m) for s, m in zip(opt_shapes, meshes))

  def init():
    stage_params = model.load(args.init_from) if args.init_from else model.init(args.seed)
    params = jax.device_put(stage_params, param_sh)
    opt_states = tuple(
        jax.jit(optimizer.init, out_shardings=s)(p) for p, s in zip(params, opt_sh)
    )
    return params, opt_states

  abstract_state = (
      tuple(abstract(s, sh) for s, sh in zip(param_shapes, param_sh)),
      tuple(abstract(s, sh) for s, sh in zip(opt_shapes, opt_sh)),
  )
  return init, abstract_state, param_sh, opt_sh


def _mpmd_layout(args, topology, model, optimizer, inputs, targets, split_batch) -> Layout:
  """A pipeline step over `model.stage_fns()` with replicated stage state.

  `inputs` and `targets` map each step input to `(ShapeDtypeStruct,
  PartitionSpec)` on the first and last stage mesh; `split_batch` turns a
  host batch into the step's inputs and targets.
  """
  import jax  # pylint: disable=g-import-not-at-top
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan import ops  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  num_stages, num_meshes = args.num_stages, len(topology)
  options = json.loads(args.mpmd_options) if args.mpmd_options else None
  config = jt_mpmd.make_config(topology, args.schedule or None, options, num_stages=num_stages)
  meshes = [topology[jt_mpmd.mesh_name(i % num_meshes)] for i in range(num_stages)]
  init, abstract_state, param_sh, opt_sh = _init_replicated(args, model, optimizer, meshes)

  in_sh, tgt_sh = _shardings(inputs, meshes[0]), _shardings(targets, meshes[-1])
  stage_fns = [ops.for_stage(fn, i) for i, fn in enumerate(model.stage_fns())]
  train_step = jt_mpmd.make_train_step(stage_fns, optimizer, max_grad_norm=args.max_grad_norm)
  scalar = NamedSharding(meshes[-1], P())
  jitted = mpmd.jit(
      train_step,
      config,
      in_shardings=(param_sh, opt_sh, in_sh, tgt_sh),
      out_shardings=(param_sh, opt_sh, scalar, scalar),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    x, t = split_batch(batch)
    return jax.device_put(x, in_sh), jax.device_put(t, tgt_sh)

  eval_step = mpmd.jit(
      functools.partial(jt_mpmd.pipeline_loss, stage_fns),
      jt_mpmd.make_config(topology, None, options, num_stages=num_stages),
      in_shardings=(param_sh, in_sh, tgt_sh),
      out_shardings=scalar,
  )
  return Layout(init, abstract_state, jitted, _shapes(inputs), _shapes(targets), put_batch,
                f"mpmd:{num_stages}",
                placement=placement(
                    topology, [jt_mpmd.mesh_name(i % num_meshes) for i in range(num_stages)]),
                eval_step=eval_step)


def _spmd_layout(args, mesh, model, optimizer, inputs, targets, split_batch) -> Layout:
  """No pipeline: every stage in one data-parallel `jax.jit` program on `mesh`.

  The step runs the stages over all microbatches at once, which gives the
  pipeline's loss and gradients when microbatches hold equal token counts, and
  updates the replicated state the way the pipeline does. `inputs`, `targets`
  and `split_batch` are `_mpmd_layout`'s: batched leaves lose their
  microbatch axis, and per-microbatch constants keep the first microbatch's.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan import ops  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  def batched(spec):
    return len(spec) > 1 and spec[1] is not None

  def flat_spec(v):
    shape, spec = v
    dims = shape.shape
    if batched(spec):
      dims = (dims[0] * dims[1],) + dims[2:]
    else:
      dims = dims[1:]
    return jax.ShapeDtypeStruct(dims, shape.dtype), P(*spec[1:])

  def flat_host(a, v):
    return a.reshape(-1, *a.shape[2:]) if batched(v[1]) else a[0]

  flat_inputs = jax.tree.map(flat_spec, inputs, is_leaf=_is_pair)
  flat_targets = jax.tree.map(flat_spec, targets, is_leaf=_is_pair)
  in_sh, tgt_sh = _shardings(flat_inputs, mesh), _shardings(flat_targets, mesh)
  init, abstract_state, param_sh, opt_sh = _init_replicated(
      args, model, optimizer, [mesh] * args.num_stages)

  stage_fns = [ops.for_stage(fn, i) for i, fn in enumerate(model.stage_fns())]
  last = len(stage_fns) - 1

  def loss_fn(stage_params, x, t):
    for i, fn in enumerate(stage_fns):
      x = fn(stage_params[i], x, t) if i == last else fn(stage_params[i], x)
    return x

  def train_step(stage_params, opt_states, x, t):
    loss, grads = jax.value_and_grad(loss_fn)(stage_params, x, t)
    norm = jt_mpmd.global_grad_norm(grads)
    updated = [
        jt_mpmd.clipped_update(optimizer, p, g, s, norm, args.max_grad_norm)
        for p, g, s in zip(stage_params, grads, opt_states)
    ]
    return (tuple(p for p, _ in updated), tuple(s for _, s in updated),
            jnp.where(jnp.isfinite(norm), loss, jnp.nan), norm)

  scalar = jax.sharding.NamedSharding(mesh, P())
  jitted = jax.jit(
      train_step,
      in_shardings=(param_sh, opt_sh, in_sh, tgt_sh),
      out_shardings=(param_sh, opt_sh, scalar, scalar),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    x, t = split_batch(batch)
    x, t = jax.tree.map(flat_host, x, inputs), jax.tree.map(flat_host, t, targets)
    return jax.device_put(x, in_sh), jax.device_put(t, tgt_sh)

  eval_step = jax.jit(loss_fn, in_shardings=(param_sh, in_sh, tgt_sh), out_shardings=scalar)
  return Layout(init, abstract_state, jitted, _shapes(flat_inputs), _shapes(flat_targets),
                put_batch, "spmd", placement=placement({"mesh": mesh}, ["mesh"] * args.num_stages),
                eval_step=eval_step)


def _routed_attention():
  """`jaxtitan.ops.attention` when the ops policy names attention, else None.

  Unrouted, a reference model keeps its own attention implementation.
  """
  from jaxtitan import ops  # pylint: disable=g-import-not-at-top

  policy = ops.current_policy()
  routed = "attention" in policy.prefs or any("attention" in s for s in policy.stages.values())
  return ops.attention if routed else None


def _maxtext(args, name, devices, optimizer):
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jaxtitan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  overrides = {
      "max_target_length": args.seq_len,
      "remat_policy": "full" if args.remat else "none",
      **_overrides(args),
  }
  cfg = mt.config(name, **overrides)
  if args.parallel == "mpmd":
    topology = _topology(args, devices, lambda d: mt.mesh(cfg, d))
    mesh = next(iter(topology.values()))
  else:
    mesh = mt.mesh(cfg, devices)
  model = mt.Model(cfg, mesh, num_stages=args.num_stages, layers=_stage_layers(args),
                   attention=_routed_attention())

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  tokens = (jax.ShapeDtypeStruct((num_mb, mb, seq), jnp.int32), P(None, "data", None))

  def split_batch(batch):
    batch = batch.reshape(num_mb, mb, seq + 1)
    return batch[..., :-1], batch[..., 1:]

  if args.parallel == "mpmd":
    layout = _mpmd_layout(args, topology, model, optimizer, tokens, tokens, split_batch)
  else:
    layout = _spmd_layout(args, mesh, model, optimizer, tokens, tokens, split_batch)
  tokens_per_step = num_mb * mb * seq
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=tokens_per_step,
      flops_per_step=model.flops_per_token(seq) * tokens_per_step,
      vocab_size=cfg.vocab_size,
  )
  return layout, work


def _maxdiffusion(args, flavor, devices, optimizer):
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jaxtitan.models.adapters import maxdiffusion as md  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  kwargs = {
      "dtype": jnp.bfloat16,
      "weights_dtype": jnp.float32,
      "remat_policy": "FULL" if args.remat else "NONE",
      **{k: tuple(v) if isinstance(v, list) else v for k, v in _overrides(args).items()},
  }
  for k in ("dtype", "weights_dtype"):
    kwargs[k] = jnp.dtype(kwargs[k])
  flux = md.transformer(flavor, **kwargs)

  def make_mesh(d):
    return jax.sharding.Mesh(
        np.asarray(d).reshape(len(d), 1), (jt_mpmd.DATA_AXIS, jt_mpmd.TENSOR_AXIS))

  model = md.Model(flux, num_stages=args.num_stages, blocks=_stage_layers(args),
                   attention=_routed_attention())

  side = args.resolution // 16
  image_tokens = side * side
  text_tokens = args.text_tokens or (512 if flux.guidance_embeds else 256)
  img_ids, txt_ids = md.position_ids(side, text_tokens)
  mb, num_mb = args.microbatch_size, args.microbatches
  b = (num_mb, mb)
  latent_shape = b + (image_tokens, flux.in_channels)
  f32 = jnp.float32
  batched = P(None, jt_mpmd.DATA_AXIS)
  inputs = {
      "img": (jax.ShapeDtypeStruct(latent_shape, f32), batched),
      "txt": (jax.ShapeDtypeStruct(b + (text_tokens, flux.joint_attention_dim), f32), batched),
      "pooled": (jax.ShapeDtypeStruct(b + (flux.pooled_projection_dim,), f32), batched),
      "timestep": (jax.ShapeDtypeStruct(b, f32), batched),
      "img_ids": (jax.ShapeDtypeStruct((num_mb,) + img_ids.shape, f32), P()),
      "txt_ids": (jax.ShapeDtypeStruct((num_mb,) + txt_ids.shape, f32), P()),
  }
  if flux.guidance_embeds:
    inputs["guidance"] = (jax.ShapeDtypeStruct(b, f32), batched)
  targets = (jax.ShapeDtypeStruct(latent_shape, f32), batched)

  def next_batch(step):
    """Noised random latents, random text embeddings, and velocity targets."""
    rng = np.random.default_rng((args.seed, step))
    latents = rng.standard_normal(latent_shape, np.float32)
    noise = rng.standard_normal(latent_shape, np.float32)
    t = rng.uniform(size=b).astype(np.float32)
    x = {
        "img": (1 - t[..., None, None]) * latents + t[..., None, None] * noise,
        "txt": rng.standard_normal(b + (text_tokens, flux.joint_attention_dim), np.float32),
        "pooled": rng.standard_normal(b + (flux.pooled_projection_dim,), np.float32),
        "timestep": t,
        "img_ids": np.broadcast_to(img_ids, (num_mb,) + img_ids.shape),
        "txt_ids": np.broadcast_to(txt_ids, (num_mb,) + txt_ids.shape),
    }
    if flux.guidance_embeds:
      x["guidance"] = np.full(b, 3.5, np.float32)
    return x, noise - latents

  if args.parallel == "mpmd":
    layout = _mpmd_layout(args, _topology(args, devices, make_mesh), model, optimizer,
                          inputs, targets, lambda batch: batch)
  else:
    layout = _spmd_layout(args, make_mesh(devices), model, optimizer, inputs, targets,
                          lambda batch: batch)
  samples = num_mb * mb
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=samples * (image_tokens + text_tokens),
      flops_per_step=model.flops_per_sample(image_tokens, text_tokens) * samples,
      next_batch=next_batch,
  )
  return layout, work


def build(args, devices, optimizer) -> tuple[Layout, Workload]:
  """The layout and workload of `--model=<provider>:<name>`."""
  provider, _, name = args.model.partition(":")
  if args.parallel == "fsdp" or args.stage_fsdp or args.tensor_parallel != 1:
    raise NotImplementedError(
        f"{args.model} trains with data parallelism only: as an MPMD pipeline with it inside "
        "stages, or as --parallel=dp"
    )
  if args.parallel == "mpmd" and args.reduce_grads != "microbatch":
    raise NotImplementedError(f"{args.model} reduces gradients per microbatch only")
  if args.init_from and provider != "maxtext":
    raise NotImplementedError(f"--init_from loads MaxText parameter checkpoints, not {provider}")
  if args.parallel == "dp" and (args.microbatches * args.microbatch_size) % len(devices):
    raise ValueError(
        f"--parallel=dp splits all {args.microbatches * args.microbatch_size} sequences of a "
        f"step across {len(devices)} devices"
    )
  if provider == "maxtext":
    return _maxtext(args, name, devices, optimizer)
  if provider == "maxdiffusion":
    return _maxdiffusion(args, name, devices, optimizer)
  raise ValueError(f"unknown model provider {provider!r}; known: {', '.join(PROVIDERS)}")
