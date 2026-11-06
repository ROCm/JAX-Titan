"""Trainer layouts for models from MaxText and MaxDiffusion.

`--model=maxtext:<name>` trains a MaxText model config (e.g.
`maxtext:llama3.1-8b`) on the token data, and `--model=maxdiffusion:<flavor>`
trains FLUX.1 (`flux-dev` or `flux-schnell`) on synthetic rectified-flow
batches until an image pipeline exists. Both run as a Shardy MPMD pipeline
with data parallelism inside each stage, reducing gradients in every
microbatch's backward pass, and attention in native JAX.
"""

import json

from jaxtitan.trainer.layout import Layout, Workload, abstract

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


def _mpmd_layout(args, topology, model, optimizer, inputs, targets, split_batch) -> Layout:
  """A pipeline step over `model.stage_fns()` with replicated stage state.

  `inputs` and `targets` map each step input to `(ShapeDtypeStruct,
  PartitionSpec)` on the first and last stage mesh; `split_batch` turns a
  host batch into the step's inputs and targets.
  """
  import jax  # pylint: disable=g-import-not-at-top
  from jaxtitan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  num_stages, num_meshes = args.num_stages, len(topology)
  options = json.loads(args.mpmd_options) if args.mpmd_options else None
  config = jt_mpmd.make_config(topology, args.schedule or None, options, num_stages=num_stages)
  meshes = [topology[jt_mpmd.mesh_name(i % num_meshes)] for i in range(num_stages)]

  def replicated(tree, mesh):
    return jax.tree.map(lambda _: NamedSharding(mesh, P()), tree)

  param_shapes = model.abstract(args.seed)
  opt_shapes = tuple(jax.eval_shape(optimizer.init, p) for p in param_shapes)
  param_sh = tuple(replicated(p, m) for p, m in zip(param_shapes, meshes))
  opt_sh = tuple(replicated(s, m) for s, m in zip(opt_shapes, meshes))

  def init():
    params = jax.device_put(model.init(args.seed), param_sh)
    opt_states = tuple(
        jax.jit(optimizer.init, out_shardings=s)(p) for p, s in zip(params, opt_sh)
    )
    return params, opt_states

  def is_pair(v):
    return isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], P)

  def shardings(specs, mesh):
    return jax.tree.map(lambda v: NamedSharding(mesh, v[1]), specs, is_leaf=is_pair)

  def shapes(specs):
    return jax.tree.map(lambda v: v[0], specs, is_leaf=is_pair)

  in_sh, tgt_sh = shardings(inputs, meshes[0]), shardings(targets, meshes[-1])
  train_step = jt_mpmd.make_train_step(model.stage_fns(), optimizer)
  jitted = mpmd.jit(
      train_step,
      config,
      in_shardings=(param_sh, opt_sh, in_sh, tgt_sh),
      out_shardings=(param_sh, opt_sh, NamedSharding(meshes[-1], P())),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    x, t = split_batch(batch)
    return jax.device_put(x, in_sh), jax.device_put(t, tgt_sh)

  abstract_state = (
      tuple(abstract(s, sh) for s, sh in zip(param_shapes, param_sh)),
      tuple(abstract(s, sh) for s, sh in zip(opt_shapes, opt_sh)),
  )
  return Layout(init, abstract_state, jitted, shapes(inputs), shapes(targets), put_batch,
                f"mpmd:{num_stages}")


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
  topology = _topology(args, devices, lambda d: mt.mesh(cfg, d))
  model = mt.Model(cfg, next(iter(topology.values())), num_stages=args.num_stages,
                   layers=_stage_layers(args))

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  tokens = (jax.ShapeDtypeStruct((num_mb, mb, seq), jnp.int32), P(None, "data", None))

  def split_batch(batch):
    batch = batch.reshape(num_mb, mb, seq + 1)
    return batch[..., :-1], batch[..., 1:]

  layout = _mpmd_layout(args, topology, model, optimizer, tokens, tokens, split_batch)
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
  topology = _topology(args, devices, lambda d: jax.sharding.Mesh(
      np.asarray(d).reshape(len(d), 1), (jt_mpmd.DATA_AXIS, jt_mpmd.TENSOR_AXIS)))
  model = md.Model(flux, num_stages=args.num_stages, blocks=_stage_layers(args))

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

  layout = _mpmd_layout(args, topology, model, optimizer, inputs, targets, lambda batch: batch)
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
  if args.parallel != "mpmd" or args.stage_fsdp or args.tensor_parallel != 1:
    raise NotImplementedError(
        f"{args.model} trains as an MPMD pipeline with data parallelism inside stages only"
    )
  if args.reduce_grads != "microbatch":
    raise NotImplementedError(f"{args.model} reduces gradients per microbatch only")
  if provider == "maxtext":
    return _maxtext(args, name, devices, optimizer)
  if provider == "maxdiffusion":
    return _maxdiffusion(args, name, devices, optimizer)
  raise ValueError(f"unknown model provider {provider!r}; known: {', '.join(PROVIDERS)}")
