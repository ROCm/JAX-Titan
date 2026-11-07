import argparse, sys, time, datetime
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax

import jaxpp.api as jaxpp


# ---------------- utils ----------------

def localize(x):
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
        B, T, D = x.shape
        h = self.n_heads
        assert D % h == 0
        Hd = D // h

        x = x.astype(self.dtype)
        qkv = nn.Dense(3 * D, use_bias=False, dtype=self.dtype, name="qkv")(x)
        qkv = qkv.reshape(B, T, 3, h, Hd)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = 1.0 / jnp.sqrt(jnp.array(Hd, dtype=self.dtype))
        attn_logits = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale

        mask = jnp.tril(jnp.ones((T, T), dtype=self.dtype))
        big_neg = jnp.array(-1e10, dtype=self.dtype)
        attn_logits = attn_logits + (1.0 - mask)[None, None, :, :] * big_neg

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
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
        y = self.ln1(x)
        y = self.attn(y)
        x = x + y
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
                self.d_model, self.n_heads, self.mlp_dim, self.dtype
            )
            for _ in range(self.n_layers)
        ]
        self.ln_f = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.lm_head = nn.Dense(self.vocab_size, dtype=self.dtype)

    def __call__(self, tokens):
        B, T = tokens.shape
        assert T <= self.max_seq_len

        x = self.token_emb(tokens)
        pos = self.pos_emb[:T].astype(self.dtype)
        x = x + pos[None, :, :]

        for lyr in self.layers:
            x = lyr(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class TinyGPT2_PP(TinyGPT2):
    """Same network, but inserts JaxPP pipeline markers at stage boundaries."""

    def __call__(self, tokens, *, pp: int):
        B, T = tokens.shape
        assert T <= self.max_seq_len
        assert pp > 0

        x = self.token_emb(tokens)
        pos = self.pos_emb[:T].astype(self.dtype)
        x = x + pos[None, :, :]

        nps = self.n_layers // pp
        assert nps * pp == self.n_layers

        for i, lyr in enumerate(self.layers, start=1):
            x = lyr(x)
            if i % nps == 0:
                x = jaxpp.pipeline_enter_stage(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ---------------- loss ----------------

def make_loss_fn(model_apply, num_ubatches: int):
    def loss_fn(params, microbatch_tokens):
        logits = model_apply(params, microbatch_tokens)  # [B, T, V]
        logits_ = logits[:, :-1, :]
        labels = microbatch_tokens[:, 1:]

        logits_f = logits_.astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits_f, axis=-1)
        labels_flat = labels.reshape(-1)
        log_probs_flat = log_probs.reshape(-1, log_probs.shape[-1])

        nll = -log_probs_flat[
            jnp.arange(labels_flat.shape[0]),
            labels_flat,
        ]
        loss = jnp.mean(nll) / num_ubatches
        return (loss, (logits_, labels))
    return loss_fn


# ---------------- SPMD data-parallel (pmap) ----------------

def build_dp_step_pmap(loss_fn, optimizer, num_devices):

    def per_device_step(opt_state, params, batch):
        # batch: [U, local_B, T]
        mu_vg = partial(jax.value_and_grad(loss_fn, has_aux=True), params)
        total_grad = None
        losses = []
        preds = []
        for ub in batch:
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
        return opt_state, params, jnp.stack(losses), jnp.stack(preds)

    pstep = jax.pmap(per_device_step, axis_name="data")

    def dp_step(opt_state_repl, params_repl, batch):
        # opt_state_repl, params_repl: replicated PyTrees over devices
        U, B, T = batch.shape
        assert B % num_devices == 0
        local_B = B // num_devices

        batch_reshaped = batch.reshape(U, num_devices, local_B, T)
        batch_sharded = jnp.swapaxes(batch_reshaped, 0, 1)  # [num_devices, U, local_B, T]

        opt_state_repl, params_repl, losses, preds = pstep(
            opt_state_repl, params_repl, batch_sharded
        )
        return opt_state_repl, params_repl, losses, preds

    return dp_step


PROFILE_START_STEP = 10
PROFILE_STOP_STEP = 20


def summarize_post_profile_times(name, times, train_steps):
    post = [t for i, t in enumerate(times) if i >= PROFILE_STOP_STEP]
    if not post:
        print(f"[{name}] no steps after profiling.")
        return
    avg = sum(post) / len(post)
    print(
        f"[{name}] avg step time per step after profiling "
        f"(steps {PROFILE_STOP_STEP}..{train_steps-1}): {avg*1000:.3f} ms"
    )


def run_spmd_dp(args, dp_step, opt_state_base, params_base, x, dtype, num_devices):
    print(f"=== SPMD data-parallel (pmap) on {num_devices} GPUs ===")

    devices = jax.devices()
    opt_state_repl = jax.device_put_replicated(opt_state_base, devices)
    params_repl = jax.device_put_replicated(params_base, devices)

    # Warmup
    opt_state_repl, params_repl, losses, _ = dp_step(opt_state_repl, params_repl, x)
    lb = float(np.array(losses[0]).sum())
    print(f"[spmd_dp warmup] loss_sum={lb:.6f}")

    times = []
    for step in range(args.train_steps):
        if step == PROFILE_START_STEP:
            print(f"{datetime.datetime.now()}: start profile (spmd_dp)")
            jax.profiler.start_trace("profile-spmd-dp")

        t0 = time.perf_counter()
        opt_state_repl, params_repl, losses, _ = dp_step(opt_state_repl, params_repl, x)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if step == PROFILE_STOP_STEP:
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), (opt_state_repl, params_repl, losses))
            print(f"{datetime.datetime.now()}: stop profile (spmd_dp)")
            jax.profiler.stop_trace()

        if (step + 1) % max(1, args.log_every) == 0:
            lb = float(np.array(losses[0]).sum())
            print(f"[spmd_dp {step+1:04d}/{args.train_steps:04d}] loss_sum={lb:.6f}")

    summarize_post_profile_times("spmd_dp", times, args.train_steps)
    return opt_state_repl, params_repl, losses


# ---------------- JaxPP PP ----------------

def build_pp_step(loss_fn, optimizer, *, mpmd_mesh, num_stages):

    @partial(jaxpp.mpmd_jit_with_loop, mpmd_mesh=mpmd_mesh)
    def pp_step(opt_state, params, batch):
        mu_vg = partial(jax.value_and_grad(loss_fn, has_aux=True), params)
        schedule = jaxpp.Std1F1B(num_stages)
        (losses, (preds, _)), grad = jaxpp.treduce(mu_vg, batch, schedule=schedule)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, losses, preds

    return pp_step


def run_jaxpp(args, pp_step, opt_state_pp, params_pp, x, dtype):
    print("=== JaxPP MPMD pipeline ===")
    opt_state_pp, params_pp, losses, _ = pp_step(opt_state_pp, params_pp, x)
    lp = float(localize(losses).sum())
    print(f"[jaxpp warmup] loss_sum={lp:.6f}")

    times = []
    for step in range(args.train_steps):
        if step == PROFILE_START_STEP:
            print(f"{datetime.datetime.now()}: start profile (jaxpp)")
            jax.profiler.start_trace("profile-jaxpp")

        t0 = time.perf_counter()
        opt_state_pp, params_pp, losses, _ = pp_step(opt_state_pp, params_pp, x)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if step == PROFILE_STOP_STEP:
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), (opt_state_pp, params_pp, losses))
            print(f"{datetime.datetime.now()}: stop profile (jaxpp)")
            jax.profiler.stop_trace()

        if (step + 1) % max(1, args.log_every) == 0:
            lp = float(localize(losses).sum())
            print(f"[jaxpp {step+1:04d}/{args.train_steps:04d}] loss_sum={lp:.6f}")

    summarize_post_profile_times("jaxpp", times, args.train_steps)
    return opt_state_pp, params_pp, losses


# ---------------- main ----------------

def main(args):
    devices = np.array(jax.devices())
    num_devices = devices.size
    assert num_devices == 8, f"expect 8 devices, found {num_devices}"

    n_layers = args.pp * args.layers_per_stage
    dtype = jnp.float32 if args.dtype == "float32" else jnp.float16

    U = args.num_ubatches
    B = args.global_batch
    T = args.seq_len
    assert B % num_devices == 0

    rng = jax.random.PRNGKey(0)
    rng, data_key = jax.random.split(rng)
    x = jax.random.randint(data_key, (U, B, T), 0, args.vocab_size)

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

    init_key = jax.random.PRNGKey(42)
    params_base = baseline_model.init(init_key, x[0])["params"]
    params_pp = pp_model.init(init_key, x[0], pp=args.pp)["params"]

    opt = optax.adam(args.lr)

    # SPMD DP setup
    if args.system in ("spmd_dp", "both"):
        dp_loss_fn = make_loss_fn(
            lambda p, mb: baseline_model.apply({"params": p}, mb),
            args.num_ubatches,
        )
        dp_step = build_dp_step_pmap(dp_loss_fn, opt, num_devices)
        opt_state_dp = opt.init(params_base)
    else:
        dp_step = None
        opt_state_dp = None

    # JaxPP PP setup
    if args.system in ("jaxpp", "both"):
        total_pp_devs = args.pp * args.dp * args.tp
        assert total_pp_devs <= num_devices
        mesh_devs = devices[:total_pp_devs].reshape(args.pp, args.dp, args.tp)
        mesh = jax.sharding.Mesh(mesh_devs, ("stages", "data", "model"))
        mpmd_mesh = jaxpp.MpmdMesh(mesh, "stages")

        pp_loss_fn = make_loss_fn(
            lambda p, mb: pp_model.apply({"params": p}, mb, pp=args.pp),
            args.num_ubatches,
        )
        pp_step = build_pp_step(pp_loss_fn, opt, mpmd_mesh=mpmd_mesh, num_stages=args.pp)
        opt_state_pp = opt.init(params_pp)
    else:
        pp_step = None
        opt_state_pp = None

    # Run systems
    if args.system == "spmd_dp":
        run_spmd_dp(args, dp_step, opt_state_dp, params_base, x, dtype, num_devices)
    elif args.system == "jaxpp":
        run_jaxpp(args, pp_step, opt_state_pp, params_pp, x, dtype)
    elif args.system == "both":
        opt_state_dp, params_dp, loss_dp = run_spmd_dp(
            args, dp_step, opt_state_dp, params_base, x, dtype, num_devices
        )
        opt_state_pp, params_pp, loss_pp = run_jaxpp(
            args, pp_step, opt_state_pp, params_pp, x, dtype
        )
        lb = float(np.array(loss_dp[0]).sum())
        lp = float(localize(loss_pp).sum())
        print(f"[parity final] spmd_dp_loss_sum={lb:.6f} | jaxpp_loss_sum={lp:.6f}")
    else:
        raise ValueError(f"Unknown system: {args.system}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        "TinyGPT2: JaxPP vs SPMD DP on one 8-GPU node"
    )
    ap.add_argument("--system",
                    choices=["spmd_dp", "jaxpp", "both"],
                    default="both")
    ap.add_argument("--pp", type=int, default=8)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--layers-per-stage", type=int, default=2)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--num-ubatches", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--mlp-dim", type=int, default=2048)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--train-steps", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    assert args.layers_per_stage > 0
    assert (args.pp * args.layers_per_stage) % args.pp == 0
    assert args.d_model % args.n_heads == 0

    sys.exit(main(args) or 0)
