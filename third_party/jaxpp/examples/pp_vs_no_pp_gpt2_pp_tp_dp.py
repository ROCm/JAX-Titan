# pp_vs_no_pp_gpt2_pp_tp_dp.py
# Compare no-PP vs PP+DP+TP losses with a tiny GPT2-style transformer.
#
# Baseline: all GPUs via SPMD data parallel (batch sharded on a single "data" axis).
# PP:       all GPUs via MPMD pipeline parallel (PP stages) with DP and TP axes
#           in the mesh layout: (stages, data, model) = (PP, DP, TP).
#
# Requires: jax, flax, optax, jaxpp  (and >= PP*DP*TP devices available)

import argparse, sys
from functools import partial
import datetime
import time

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax

import jaxpp.api as jaxpp


# ---------------- utils ----------------

def localize(x):
    """Return a host ndarray for both regular JAX arrays and JaxPP MpmdArray."""
    if hasattr(x, "to_mpmd_local_array"):
        return np.array(x.to_mpmd_local_array)
    return np.array(x)


# ---------------- tiny GPT2-style model ----------------

class CausalSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        h = self.n_heads
        assert D % h == 0, "d_model must be divisible by n_heads"
        head_dim = D // h

        x = x.astype(self.dtype)

        # Project to qkv
        qkv = nn.Dense(3 * D, use_bias=False, dtype=self.dtype, name="qkv")(x)
        qkv = qkv.reshape(B, T, 3, h, head_dim)
        q = qkv[:, :, 0]  # [B, T, h, Hd]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # [B, h, T, Hd]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = 1.0 / jnp.sqrt(jnp.array(head_dim, dtype=self.dtype))
        attn_logits = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale  # [B, h, T, T]

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=self.dtype))
        big_neg = jnp.array(-1e10, dtype=self.dtype)
        attn_logits = attn_logits + (1.0 - mask)[None, None, :, :] * big_neg

        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn = jnp.matmul(attn_weights, v)  # [B, h, T, Hd]

        attn = jnp.transpose(attn, (0, 2, 1, 3))  # [B, T, h, Hd]
        attn = attn.reshape(B, T, D)

        out = nn.Dense(D, dtype=self.dtype, name="out")(attn)
        return out


class MLP(nn.Module):
    d_model: int
    mlp_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.mlp_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, dtype=self.dtype)(x)
        return x


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.ln1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.ln2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.attn = CausalSelfAttention(self.d_model, self.n_heads, self.dtype)
        self.mlp = MLP(self.d_model, self.mlp_dim, self.dtype)

    def __call__(self, x):
        # Self-attention + residual
        y = self.ln1(x)
        y = self.attn(y)
        x = x + y

        # MLP + residual
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + y
        return x


class TinyGPT2(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    mlp_dim: int
    max_seq_len: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.token_emb = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            dtype=self.dtype,
        )
        self.pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, self.d_model),
        )
        self.layers = [
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype,
            )
            for _ in range(self.n_layers)
        ]
        self.ln_f = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.lm_head = nn.Dense(self.vocab_size, dtype=self.dtype)

    def __call__(self, tokens):
        # tokens: [B, T], int32
        B, T = tokens.shape
        assert T <= self.max_seq_len, "sequence length exceeds max_seq_len"

        x = self.token_emb(tokens)  # [B, T, D]
        pos = self.pos_emb[:T].astype(self.dtype)  # [T, D]
        x = x + pos[None, :, :]

        for lyr in self.layers:
            x = lyr(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits


class TinyGPT2_PP(TinyGPT2):
    """Same network, but inserts pipeline markers at stage boundaries."""

    def __call__(self, tokens, *, pp: int):
        B, T = tokens.shape
        assert T <= self.max_seq_len
        assert pp > 0

        x = self.token_emb(tokens)
        pos = self.pos_emb[:T].astype(self.dtype)
        x = x + pos[None, :, :]

        nps = self.n_layers // pp
        assert nps * pp == self.n_layers, "n_layers must be divisible by pp"

        for i, lyr in enumerate(self.layers, start=1):
            x = lyr(x)
            # Insert exactly `pp` markers (one per stage end).
            if i % nps == 0:
                x = jaxpp.pipeline_enter_stage(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ---------------- losses ----------------

def make_loss_fn(model_apply, num_ubatches: int):
    """Cross-entropy loss for next-token prediction.

    loss_fn(params, microbatch_tokens[ B, T ]) -> (scalar_loss, (logits, labels))
    """
    def loss_fn(params, microbatch_tokens):
        logits = model_apply(params, microbatch_tokens)  # [B, T, V]

        # Next-token prediction: predict token t+1 from token t
        logits_ = logits[:, :-1, :]          # [B, T-1, V]
        labels = microbatch_tokens[:, 1:]    # [B, T-1]

        # Work in float32 for stability even if model is fp16
        logits_f = logits_.astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits_f, axis=-1)  # [B, T-1, V]
        labels_flat = labels.reshape(-1)
        log_probs_flat = log_probs.reshape(-1, log_probs.shape[-1])

        nll = -log_probs_flat[
            jnp.arange(labels_flat.shape[0]),
            labels_flat,
        ]  # [B*(T-1)]
        loss = jnp.mean(nll) / num_ubatches

        return (loss, (logits_, labels))

    return loss_fn


# ---------------- separate train steps ----------------

def build_base_train_step_spmd(loss_fn, optimizer, *, spmd_mesh):
    """Baseline using SPMD data parallelism on the batch dimension.

    step(opt_state, params, batch[U,B,T])
      -> (opt_state, params, losses[U], preds[U,B,T,V])

    - spmd_mesh: jax.sharding.Mesh over `N = PP*DP*TP` devices with axis name 'data'.
    - We shard batch as PartitionSpec(None, 'data', None) => shard B across devices.
    """
    P = jax.sharding.PartitionSpec
    batch_sharding = jax.sharding.NamedSharding(spmd_mesh, P(None, "data", None))

    @partial(
        jax.jit,
        in_shardings=(None, None, batch_sharding),
        out_shardings=(None, None, None, None),  # gather results on host
    )
    def base_step(opt_state, params, batch):
        mu_vg = partial(jax.value_and_grad(loss_fn, has_aux=True), params)

        total_grad = None
        losses, preds = [], []
        # batch: [U, B, T], with B sharded on 'data'
        for ub in batch:  # ub: [B, T], still sharded
            (l, (p, _)), g = mu_vg(ub)
            losses.append(l)
            preds.append(p)
            total_grad = (
                g
                if total_grad is None
                else jax.tree_util.tree_map(jnp.add, total_grad, g)
            )

        updates, opt_state = optimizer.update(total_grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, jnp.asarray(losses), jnp.asarray(preds)

    return base_step


def build_pp_train_step(loss_fn, optimizer, *, mpmd_mesh, num_stages: int):
    """Pipeline-parallel version using JaxPP.

    step(opt_state, params, batch[U,B,T])
      -> (opt_state, params, losses[U], preds[U,B,T,V])
    """

    @partial(jaxpp.mpmd_jit_with_loop, mpmd_mesh=mpmd_mesh)
    def pp_step(opt_state, params, batch):
        mu_vg = partial(jax.value_and_grad(loss_fn, has_aux=True), params)
        schedule = jaxpp.Std1F1B(num_stages)
        (losses, (preds, _)), grad = jaxpp.treduce(mu_vg, batch, schedule=schedule)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, losses, preds

    return pp_step


# ---------------- main loops ----------------

# Zero-based step indices
PROFILE_START_STEP = 10  # when to start profiling
PROFILE_STOP_STEP = 20   # when to stop profiling (inclusive)


def summarize_post_profile_times(name, step_times, train_steps):
    """Report average step time *after* profiling has completed.

    For default PROFILE_STOP_STEP=20 and train_steps=40,
    this is steps 20..39 (inclusive).
    """
    post_profile_times = [
        t for i, t in enumerate(step_times) if i >= PROFILE_STOP_STEP
    ]
    if not post_profile_times:
        print(f"[{name}] no steps after profiling to summarize.")
        return
    avg = sum(post_profile_times) / len(post_profile_times)
    print(
        f"[{name}] avg step time per step after profiling "
        f"(steps {PROFILE_STOP_STEP}..{train_steps-1}): {avg*1000:.3f} ms"
    )


def run_baseline(args, base_step, opt_state_base, params_base, x, dtype):
    print("=== Running baseline (SPMD data parallel, no PP) ===")
    # Warmup (not timed)
    opt_state_base, params_base, loss_base, _ = base_step(opt_state_base, params_base, x)
    lb = float(localize(loss_base).sum())
    print(f"[baseline warmup] loss_sum={lb:.6f}")

    step_times = []

    # Train
    for step in range(args.train_steps):
        if step == PROFILE_START_STEP:
            print(f"{datetime.datetime.now()}: starting profile (baseline)")
            jax.profiler.start_trace("test-baseline-no-pp-profile")

        t0 = time.perf_counter()
        opt_state_base, params_base, loss_base, _ = base_step(opt_state_base, params_base, x)
        jax.block_until_ready(loss_base)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        if step == PROFILE_STOP_STEP:
            print(f"{datetime.datetime.now()}: blocking (baseline)")
            jax.block_until_ready(opt_state_base)
            print(f"{datetime.datetime.now()}: stopping profile (baseline)")
            jax.profiler.stop_trace()

        if (step + 1) % max(1, args.log_every) == 0:
            lb = float(localize(loss_base).sum())
            print(f"[baseline {step+1:04d}/{args.train_steps:04d}] loss_sum={lb:.6f}")

    summarize_post_profile_times("baseline", step_times, args.train_steps)
    return opt_state_base, params_base, loss_base


def run_pipeline(args, pp_step, opt_state_pp, params_pp, x, dtype):
    print("=== Running pipeline-parallel (PP+DP+TP MPMD) ===")
    # Warmup (not timed)
    opt_state_pp, params_pp, loss_pp, _ = pp_step(opt_state_pp, params_pp, x)
    lp = float(localize(loss_pp).sum())
    print(f"[pp warmup] loss_sum={lp:.6f}")

    step_times = []

    # Train
    for step in range(args.train_steps):
        if step == PROFILE_START_STEP:
            print(f"{datetime.datetime.now()}: starting profile (PP)")
            jax.profiler.start_trace("test-pp-profile")

        t0 = time.perf_counter()
        opt_state_pp, params_pp, loss_pp, _ = pp_step(opt_state_pp, params_pp, x)
        jax.block_until_ready(loss_pp)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        if step == PROFILE_STOP_STEP:
            print(f"{datetime.datetime.now()}: blocking (PP)")
            jax.block_until_ready(opt_state_pp)
            print(f"{datetime.datetime.now()}: stopping profile (PP)")
            jax.profiler.stop_trace()

        if (step + 1) % max(1, args.log_every) == 0:
            lp = float(localize(loss_pp).sum())
            print(f"[pp {step+1:04d}/{args.train_steps:04d}] loss_sum={lp:.6f}")

    summarize_post_profile_times("pp", step_times, args.train_steps)
    return opt_state_pp, params_pp, loss_pp


# ---------------- main ----------------

def main(args):
    # Parallelism config
    pp = args.pp
    dp = args.dp
    tp = args.tp
    total_devices_needed = pp * dp * tp

    devices = np.array(jax.devices())
    assert devices.size >= total_devices_needed, (
        f"Need at least {total_devices_needed} devices, got {devices.size}"
    )

    print(
        f"Config: PP={pp}, DP={dp}, TP={tp}, "
        f"total_devices={total_devices_needed}, "
        f"train_steps={args.train_steps}"
    )

    # Shared model config
    n_layers = pp * args.layers_per_stage
    dtype = jnp.float32 if args.dtype == "float32" else jnp.float16

    # Synthetic token data: batch of U microbatches, each [B, T]
    U = args.num_ubatches
    B = args.global_batch
    T = args.seq_len

    # For clean data parallel sharding, ensure B divisible by total_devices_needed
    assert B % total_devices_needed == 0, (
        f"global_batch ({B}) must be divisible by total_devices "
        f"PP*DP*TP={total_devices_needed}"
    )

    rng = jax.random.PRNGKey(0)
    rng, data_key, init_key = jax.random.split(rng, 3)
    x = jax.random.randint(data_key, (U, B, T), 0, args.vocab_size)

    # Models
    baseline_model = TinyGPT2(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=n_layers,
        n_heads=args.n_heads,
        mlp_dim=args.mlp_dim,
        max_seq_len=args.seq_len,
        dtype=dtype,
    )
    pp_model = TinyGPT2_PP(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=n_layers,
        n_heads=args.n_heads,
        mlp_dim=args.mlp_dim,
        max_seq_len=args.seq_len,
        dtype=dtype,
    )

    # Init params with *same* key for parity when both are used
    params_base = baseline_model.init(init_key, x[0])
    params_pp = pp_model.init(init_key, x[0], pp=pp)

    # Optimizer
    opt = optax.adam(args.lr)

    # Baseline loss fn
    base_loss_fn = make_loss_fn(
        lambda p, mb: baseline_model.apply(p, mb),
        args.num_ubatches,
    )

    # Build SPMD mesh for baseline if needed
    if args.mode in ("base", "both"):
        # 1D mesh over all devices for data parallel
        mesh_spmd = jax.sharding.Mesh(devices[:total_devices_needed], ("data",))
        base_step = build_base_train_step_spmd(base_loss_fn, opt, spmd_mesh=mesh_spmd)
        opt_state_base = opt.init(params_base)
    else:
        base_step = None
        opt_state_base = None

    # Build MpmdMesh & PP step if needed
    if args.mode in ("pp", "both"):
        mesh_devices = devices[:total_devices_needed].reshape(pp, dp, tp)
        mesh = jax.sharding.Mesh(
            mesh_devices,
            ("stages", "data", "model"),
        )
        mpmd_mesh = jaxpp.MpmdMesh(mesh, "stages")

        pp_loss_fn = make_loss_fn(
            lambda p, mb: pp_model.apply(p, mb, pp=pp),
            args.num_ubatches,
        )
        pp_step = build_pp_train_step(
            pp_loss_fn,
            opt,
            mpmd_mesh=mpmd_mesh,
            num_stages=pp,
        )
        opt_state_pp = opt.init(params_pp)
    else:
        mpmd_mesh = None
        pp_step = None
        opt_state_pp = None

    # ---------- Run according to mode ----------

    if args.mode in ("base", "both"):
        opt_state_base, params_base, loss_base = run_baseline(
            args, base_step, opt_state_base, params_base, x, dtype
        )

    if args.mode in ("pp", "both"):
        opt_state_pp, params_pp, loss_pp = run_pipeline(
            args, pp_step, opt_state_pp, params_pp, x, dtype
        )

    # Optional parity check when we did both
    if args.mode == "both":
        lb = float(localize(loss_base).sum())
        lp = float(localize(loss_pp).sum())
        print(f"[parity final] noPP_loss_sum={lb:.6f} | PP_loss_sum={lp:.6f}")
        np.testing.assert_allclose(
            lb,
            lp,
            rtol=1e-3 if dtype == jnp.float32 else 5e-3,
            atol=1e-4,
        )
        print("SUCCESS: PP matches baseline within tolerance.")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        "Minimal: compare / profile SPMD vs PP+DP+TP with Tiny GPT2 + JaxPP"
    )
    ap.add_argument("--mode", type=str, choices=["base", "pp", "both"], default="both",
                    help="Run baseline only, PP only, or both (with parity check).")
    ap.add_argument("--pp", type=int, default=8,
                    help="pipeline stages (PP)")
    ap.add_argument("--dp", type=int, default=1,
                    help="data parallel factor (DP)")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor parallel factor (TP)")
    ap.add_argument("--layers-per-stage", type=int, default=2,
                    help="transformer layers per pipeline stage")
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--num-ubatches", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--mlp-dim", type=int, default=2048)
    ap.add_argument("--dtype", type=str, choices=["float32", "float16"], default="float32")
    ap.add_argument("--train-steps", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    # Basic sanity checks
    assert args.layers_per_stage > 0
    assert (args.pp * args.layers_per_stage) % args.pp == 0
    assert args.d_model % args.n_heads == 0

    sys.exit(main(args) or 0)
