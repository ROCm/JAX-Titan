"""Pipeline stages of MaxText decoder-only models, pinned at maxtext-v0.2.5.

MaxText supplies the model configs, the layers and the parameter layout;
JAX-Titan supplies the stages, meshes and train step. Stage 0 holds MaxText's
token embedder, the last stage its final norm and output head, and every stage
a contiguous slice of the scanned decoder layers. Stage parameters are plain
dicts with MaxText's parameter names, and stacked layer weights keep their
layer axis where MaxText puts it (`param_scan_axis`), so they map one to one
onto MaxText checkpoints.

Supported models are those MaxText builds as one scanned stack of identical
layers with an untied output head, such as Llama 2 and 3. MaxText's layers
constrain activation shardings, so stages reduce gradients per microbatch.
`llama_config` and `llama_params` map such a model onto
`jaxtitan.models.llama`.
"""

from collections.abc import Callable, Sequence
import functools
import os

# MaxText reaches for Google Cloud clients unless this is set.
os.environ.setdefault("DECOUPLE_GCLOUD", "TRUE")

# pylint: disable=g-import-not-at-top,wrong-import-position
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from maxtext.common.common_types import DecoderBlockType
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import nnx_scan
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.utils.sharding import create_sharding

from jaxtitan import paths
from jaxtitan.models import llama

PINNED_VERSION = "maxtext-v0.2.5"
_LOGITS_AXES = ("activation_embed_and_logits_batch", "activation_length", "activation_vocab")
_SCANNED_GENERIC = (DecoderBlockType.LLAMA2,)


def config(model_name: str, **overrides):
  """The MaxText config of `configs/models/<model_name>.yml`, with overrides.

  Defaults suit JAX-Titan: no dropout, XLA dot-product attention, no
  checkpointing or distributed setup by MaxText. Overriding a size such as
  `base_num_decoder_layers` also sets `override_model_config`.
  """
  kwargs = {
      "run_name": "jaxtitan",
      "base_output_directory": str(paths.work_dir() / "maxtext"),
      "skip_jax_distributed_system": True,
      "enable_checkpointing": False,
      "enable_dropout": False,
      "attention": "dot_product",
  }
  kwargs.update(overrides)
  if any(k.startswith("base_") or k in ("head_dim", "vocab_size") for k in overrides):
    kwargs.setdefault("override_model_config", True)
  base = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
  return pyconfig.initialize(["", base], model_name=model_name, **kwargs)


def mesh(cfg, devices: Sequence[jax.Device]) -> jax.sharding.Mesh:
  """A mesh with MaxText's axis names and every device on the data axis."""
  shape = [1] * len(cfg.mesh_axes)
  shape[cfg.mesh_axes.index("data")] = len(devices)
  return jax.sharding.Mesh(np.asarray(devices).reshape(shape), cfg.mesh_axes)


def _interleaved_rope(cfg) -> bool:
  """Whether MaxText rotates adjacent pairs, as its Llama 3.1 RoPE does."""
  return cfg.model_name.startswith("llama3.1") or cfg.rope_type.lower().startswith("llama3.1")


def llama_config(cfg) -> llama.LlamaConfig:
  """The same model as a `jaxtitan.models.llama` config."""
  interleaved = _interleaved_rope(cfg)
  if cfg.rope_min_timescale != 1 or not interleaved and (
      cfg.rope_type.lower() != "default" or cfg.rope_linear_scaling_factor != 1.0):
    raise NotImplementedError(
        f"rope_type={cfg.rope_type}, rope_min_timescale={cfg.rope_min_timescale}, "
        f"rope_linear_scaling_factor={cfg.rope_linear_scaling_factor}"
    )
  return llama.LlamaConfig(
      vocab_size=cfg.vocab_size,
      emb_dim=cfg.emb_dim,
      num_layers=cfg.num_decoder_layers,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      mlp_dim=cfg.mlp_dim,
      norm_eps=cfg.normalization_layer_epsilon,
      rope_theta=float(cfg.rope_max_timescale),
      rope_scaling=llama.Llama31RopeScaling() if interleaved and cfg.rope_use_scale else None,
  )


def llama_params(cfg, stage_params: Sequence[dict]) -> tuple[dict, ...]:
  """Stage parameters in the `jaxtitan.models.llama` layout.

  Puts the layer axis first, takes MaxText's 1/sqrt(head_dim) attention scale
  back out of the query kernel, and, for interleaved RoPE, reorders the query
  and key features of each head into split halves.
  """
  def first(x):
    return jnp.moveaxis(x, cfg.param_scan_axis, 0)

  def rope(x):
    return jnp.concatenate([x[..., ::2], x[..., 1::2]], axis=-1) if _interleaved_rope(cfg) else x

  out = []
  for p in stage_params:
    attn, mlp = p["layers"]["self_attention"], p["layers"]["mlp"]
    q = rope(first(attn["query"]["kernel"]))
    q = (q.astype(jnp.float32) * np.sqrt(cfg.head_dim).astype(np.float32)).astype(q.dtype)
    k = rope(first(attn["key"]["kernel"]))
    v = first(attn["value"]["kernel"])
    o = first(attn["out"]["kernel"])
    n, e = q.shape[0], q.shape[1]
    stage = {"layers": {
        "attn_norm": first(p["layers"]["pre_self_attention_layer_norm"]["scale"]),
        "wq": q.reshape(n, e, -1),
        "wk": k.reshape(n, e, -1),
        "wv": v.reshape(n, e, -1),
        "wo": o.reshape(n, -1, e),
        "mlp_norm": first(p["layers"]["post_self_attention_layer_norm"]["scale"]),
        "w_gate": first(mlp["wi_0"]["kernel"]),
        "w_up": first(mlp["wi_1"]["kernel"]),
        "w_down": first(mlp["wo"]["kernel"]),
    }}
    if "token_embedder" in p:
      stage["embed"] = p["token_embedder"]["embedding"]
    if "decoder_norm" in p:
      stage["final_norm"] = p["decoder_norm"]["scale"]
      stage["lm_head"] = p["logits_dense"]["kernel"]
    out.append(stage)
  return tuple(out)


def _slice_layers(x, start: int, size: int, scan_axis: int):
  axis = scan_axis if x.ndim > scan_axis else 0
  return jax.lax.slice_in_dim(x, start, start + size, axis=axis)


class Model:
  """A MaxText decoder-only model split into pipeline stages.

  `mesh` is any one stage mesh from `mesh()`: MaxText's layers place their
  sharding constraints on it, and an MPMD program needs every constraint on
  one mesh. `layers` gives the decoder layers per stage; the default splits
  them evenly.
  """

  def __init__(self, cfg, stage_mesh: jax.sharding.Mesh, *, num_stages: int,
               layers: Sequence[int] | None = None):
    if cfg.decoder_block not in _SCANNED_GENERIC or not cfg.scan_layers:
      raise NotImplementedError(
          f"decoder_block={cfg.decoder_block} with scan_layers={cfg.scan_layers}; "
          f"supported: {[b.value for b in _SCANNED_GENERIC]} with scan_layers=True"
      )
    if cfg.logits_via_embedding:
      raise NotImplementedError("tied input and output embeddings span the first and last stage")
    if cfg.remat_policy not in ("full", "none"):
      raise NotImplementedError(f"remat_policy={cfg.remat_policy}; supported: full, none")
    if layers is None:
      if cfg.num_decoder_layers % num_stages:
        raise ValueError(f"{cfg.num_decoder_layers} layers do not split into {num_stages} stages")
      layers = (cfg.num_decoder_layers // num_stages,) * num_stages
    if len(layers) != num_stages or sum(layers) != cfg.num_decoder_layers:
      raise ValueError(f"stage layers {tuple(layers)} do not split {cfg.num_decoder_layers} layers")
    self.cfg = cfg
    self.mesh = stage_mesh
    self.layers = tuple(layers)
    self.num_stages = num_stages

    with nn.logical_axis_rules(cfg.logical_axis_rules):
      abstract = nnx.eval_shape(lambda: self._transformer(0))
    self._embed_graph = nnx.graphdef(abstract.token_embedder)
    self._layers_graph = nnx.split(abstract.decoder.layers, nnx.Param, ...)[0]
    self._norm_graph = nnx.graphdef(abstract.decoder.decoder_norm)
    self._logits_graph = nnx.graphdef(abstract.decoder.logits_dense)
    rest = jax.jit(lambda: self._init_full(0)[1])()
    self._layer_rest = self._split_layers(rest["decoder"]["layers"])

  def _transformer(self, seed: int):
    return models.Transformer(
        self.cfg, self.mesh, quant=None, rngs=nnx.Rngs(params=seed, dropout=seed)
    )

  def _init_full(self, seed: int):
    with nn.logical_axis_rules(self.cfg.logical_axis_rules):
      _, params, rest = nnx.split(self._transformer(seed), nnx.Param, ...)
    return params.to_pure_dict(), rest.to_pure_dict()

  def _split_layers(self, stacked) -> list:
    out, start = [], 0
    for n in self.layers:
      out.append(jax.tree.map(
          functools.partial(_slice_layers, start=start, size=n, scan_axis=self.cfg.param_scan_axis),
          stacked,
      ))
      start += n
    return out

  def split(self, params: dict) -> tuple[dict, ...]:
    """Stage parameter dicts from the parameters of MaxText's `Transformer`."""
    decoder = params["decoder"]
    stages = [{"layers": p} for p in self._split_layers(decoder["layers"])]
    stages[0]["token_embedder"] = params["token_embedder"]
    stages[-1]["decoder_norm"] = decoder["decoder_norm"]
    stages[-1]["logits_dense"] = decoder["logits_dense"]
    return tuple(stages)

  def init(self, seed: int = 0) -> tuple[dict, ...]:
    """Stage parameters, initialized as MaxText initializes `Transformer`."""
    return self.split(jax.jit(lambda: self._init_full(seed)[0])())

  def abstract(self, seed: int = 0) -> tuple[dict, ...]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return jax.eval_shape(lambda: self.split(self._init_full(seed)[0]))

  def forward(self, stage: int, params: dict, x: jax.Array) -> jax.Array:
    """Token ids in for stage 0; logits out of the last stage."""
    cfg = self.cfg
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      if stage == 0:
        embedder = nnx.merge(self._embed_graph, params["token_embedder"])
        y = embedder(x.astype(jnp.int32), model_mode=MODEL_MODE_TRAIN).astype(cfg.dtype)
      else:
        y = x
      batch, length = y.shape[0], y.shape[1]
      positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
      segments = jnp.ones((batch, length), jnp.int32)
      layers = nnx.merge(self._layers_graph, params["layers"], self._layer_rest[stage])
      y = nnx_scan.apply_scanned_layers(
          layers, y,
          length=self.layers[stage],
          param_scan_axis=cfg.param_scan_axis,
          apply_fn=lambda layer, h: layer(h, segments, positions, True, MODEL_MODE_TRAIN)[0],
          remat=cfg.remat_policy != "none",
          remat_policy=None,
          prevent_cse=maxtext_utils.should_prevent_cse_in_remat(cfg),
      )
      if stage < self.num_stages - 1:
        return y
      y = nnx.merge(self._norm_graph, params["decoder_norm"])(y, out_sharding=None)
      logits = nnx.merge(self._logits_graph, params["logits_dense"])(
          y, out_sharding=create_sharding(self.mesh, _LOGITS_AXES)
      )
      return logits.astype(jnp.float32) if cfg.cast_logits_to_fp32 else logits

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """Stage functions for `jaxtitan.mpmd.make_train_step`.

    Stage `i` is called as `fn(params, x)`; the last stage also takes the
    targets and returns the mean next-token cross-entropy.
    """
    fns = [functools.partial(self.forward, i) for i in range(self.num_stages - 1)]
    last = self.num_stages - 1
    fns.append(lambda p, x, t: llama.cross_entropy(self.forward(last, p, x), t))
    return fns

  def flops_per_token(self, seq_len: int) -> float:
    """Training FLOPs per token: 6N for the weights plus causal attention."""
    cfg = self.cfg
    e, f, d = cfg.emb_dim, cfg.mlp_dim, cfg.head_dim
    per_layer = e * d * (2 * cfg.num_query_heads + 2 * cfg.num_kv_heads) + 3 * e * f
    attn = 6 * cfg.num_decoder_layers * cfg.num_query_heads * d * seq_len
    return 6 * (cfg.num_decoder_layers * per_layer + cfg.vocab_size * e) + attn
