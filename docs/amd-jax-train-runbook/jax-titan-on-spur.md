# JAX-Titan on one SPUR node

This runbook trains JAX-Titan on one SPUR node with 8 AMD Instinct MI355X
GPUs: get a node, start the TheRock container on it, build or reuse the JAX
wheels, set up the venv and the data, run the CPU tests, train, checkpoint,
profile, and release the node. It wraps the commands of
[getting_started.md](../getting_started.md) in the SPUR procedures of
[spur-cluster-guide.md](spur-cluster-guide.md); read that guide's rules and
Cluster facts first.

> **Status.** The JAX-Titan commands are the ones in getting_started.md and
> pipelines.md, tested on a shared 8x MI355X node outside SPUR. The SPUR
> procedures (allocating, `spur exec`, renewing, releasing) were tested on the
> v2 pool on 2026-09-16 by the source runbooks, with other software.
> **Nobody has run JAX-Titan on SPUR yet.** The adapted container script, the
> helper scripts, the storage layout and the task-mode script below are
> derived from both and are untested. If something here is wrong, fix this
> file.

[pipelines.md](../pipelines.md) describes the schedules and layouts and has
the measured results; this runbook does not repeat them.

## 1. Where things live

The container mounts the directory that holds the JAX-Titan checkout, so keep
the checkout, a MaxText checkout (for the Llama 3 tokenizer) and
`JAX_TITAN_ROOT` side by side on the shared home. Only the Bazel output root
and throwaway checkpoints go to node-local scratch.

| What | Where | Why |
|---|---|---|
| JAX-Titan and MaxText checkouts | Shared home, side by side | Visible on the login node and on every node |
| `JAX_TITAN_ROOT` (`.jax_titan` next to the checkout): source trees, wheels, venv, data, compilation cache, logs, profiles, checkpoints you keep | Shared home | Built once and reused on any node; logs readable from the login node |
| Bazel output root | Node-local scratch | Large, and only needed while building |
| Throwaway checkpoints | Node-local scratch or `/dev/shm` | Fast, but gone with the node |

On the development machine, the source trees, venv and data took about 3 GB,
the wheels 189 MB and the Bazel output root about 60 GB. A Llama 3.1 8B
checkpoint is 80 GB. The shared home is a group-wide volume with limited free
space (Cluster facts), so check `df -h "$HOME"` before saving checkpoints
there.

## 2. One-time setup on the login node

Set the variables. Values in `<>` come from Cluster facts in
spur-cluster-guide.md:

```bash
SPUR=<v2 controller URL>
PART=<v2 partition>
NODE=<v2 compute node to request>
SCRATCH=<node-local scratch path>

JT_REPO=$HOME/jax_titan/jax-titan          # the JAX-Titan checkout
JT_ROOT=$(dirname "$JT_REPO")/.jax_titan    # JAX-Titan's default JAX_TITAN_ROOT
JT_SPUR=$JT_ROOT/spur                      # settings and scripts for SPUR
JT_SCRATCH=$SCRATCH/$USER/jax_titan         # node-local
```

Clone JAX-Titan and MaxText side by side. The tokenizer is read from the
MaxText checkout next to JAX-Titan (override with `JAX_TITAN_LLAMA3_TOKENIZER`).

```bash
mkdir -p "$(dirname "$JT_REPO")" && cd "$(dirname "$JT_REPO")"
git clone <JAX-Titan repository URL> jax-titan
git clone https://github.com/AI-Hypercomputer/maxtext.git maxtext
```

Then create the `jax`, `xla` and `shardy` trees in `$JT_ROOT/src` and apply
the patches, as in getting_started.md section 2. These are git operations and
can run on the login node.

### Settings

Write the settings file. It records your login-node user, uid and gid,
because `spur exec` may not run as you (section 4):

```bash
mkdir -p "$JT_SPUR" "$JT_ROOT/logs"
cat > "$JT_SPUR/spur.env" <<EOF
# JAX-Titan on SPUR. Sourced on the login node, on the node and in the container.
export SPUR='$SPUR'
export SCRATCH='$SCRATCH'
export JT_REPO='$JT_REPO'
export JT_ROOT='$JT_ROOT'
export JT_SPUR='$JT_SPUR'
export JT_SCRATCH='$JT_SCRATCH'
export JT_USER='$USER' JT_UID='$(id -u)' JT_GID='$(id -g)' JT_GROUP='$(id -gn)'
export JT_CONTAINER='${USER}_therock-10.1_jax_titan'
export JT_IMAGE='ghcr.io/rocm/jax-base-ubu24.therock-10.1:latest'
export DOCKER_CONFIG=/tmp/.docker
EOF
cat >> "$JT_SPUR/spur.env" <<'EOF'
mkdir -p "$DOCKER_CONFIG" 2>/dev/null || true
# Login-node helpers; set JOBID first. on_node and in_jt pass their arguments
# to a shell on the node as one string; in_jt also passes on TRAIN_TIMEOUT.
on_node() { spur exec --controller "$SPUR" "$JOBID" bash -c "source $JT_SPUR/spur.env && $*"; }
in_jt() { on_node "${TRAIN_TIMEOUT:+TRAIN_TIMEOUT=$TRAIN_TIMEOUT }$JT_SPUR/jt.sh $*"; }
# Succeeds only if scripts/gpus_free.sh printed "free" on the node.
gpus_free() { local out; out=$(on_node '$JT_REPO/scripts/gpus_free.sh'); echo "$out"; grep -qx free <<< "$out"; }
EOF
```

`on_node` runs a command on the node, outside the container. `in_jt` runs a
command in the JAX-Titan container as you, from the checkout; `in_jt -d`
detaches, and `TRAIN_TIMEOUT=30m in_jt ...` gives train.sh a time limit.
Simple commands pass through unchanged. Put a compound command in
one double-quoted string with single quotes inside, for example
`in_jt "bash -c 'id; nproc'"`, or in a script on the shared home.

`gpus_free` runs `scripts/gpus_free.sh` on the node and checks its output
rather than its exit status, because nobody has checked whether `spur exec`
passes on the remote exit status. `gpus_free && <run>` starts the run only
when the GPUs are free.

### Scripts that run on the node

`container.sh` is `scripts/container.sh` adapted for SPUR (section 5 lists the
differences). `jt.sh` runs a command in the container.

```bash
cat > "$JT_SPUR/container.sh" <<'EOF'
#!/usr/bin/env bash
# scripts/container.sh for a SPUR node. Starts the JAX-Titan container with all
# GPUs, mounts the checkout's parent directory and node-local scratch, and adds
# a user with your login-node uid and gid, so files keep your owner. Reuses the
# container if it exists. Runs on the node, outside any container.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/spur.env"
mkdir -p "$DOCKER_CONFIG"
mkdir -p "$JT_SCRATCH" || true

if docker container inspect "$JT_CONTAINER" >/dev/null 2>&1; then
  docker start "$JT_CONTAINER" >/dev/null
  echo "reusing ${JT_CONTAINER} on $(hostname)"
  exit 0
fi

WORKSPACE=$(dirname "$JT_REPO")
VIDEO_GID=$(getent group video | cut -d: -f3) || { echo "no video group on $(hostname)" >&2; exit 1; }
RENDER_GID=$(getent group render | cut -d: -f3) || { echo "no render group on $(hostname)" >&2; exit 1; }

docker run -d --name "$JT_CONTAINER" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
  --ipc=host --shm-size=64G --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE --network=host \
  -v "$WORKSPACE:$WORKSPACE" -v "$JT_SCRATCH:$JT_SCRATCH" -w "$JT_REPO" \
  "$JT_IMAGE" sleep infinity

docker exec "$JT_CONTAINER" bash -lc "
  getent group $VIDEO_GID || groupadd -g $VIDEO_GID video
  getent group $RENDER_GID || groupadd -g $RENDER_GID render
  getent group $JT_GID || groupadd -g $JT_GID $JT_GROUP
  useradd -u $JT_UID -g $JT_GID -G $VIDEO_GID,$RENDER_GID \
    -d /home/$JT_USER -M -s /bin/bash $JT_USER
  mkdir -p /home/$JT_USER && chown $JT_UID:$JT_GID /home/$JT_USER
  chown $JT_UID:$JT_GID $JT_SCRATCH"
echo "started ${JT_CONTAINER} on $(hostname)"
EOF

cat > "$JT_SPUR/jt.sh" <<'EOF'
#!/usr/bin/env bash
# Runs a command in the JAX-Titan container as you, from the checkout. Runs on
# the node, outside the container. -d detaches, so the command keeps running
# after the spur exec or srun call that started it returns. docker exec sets
# no USER, which scripts/cleanup_after_train.sh needs.
#   jt.sh [-d] COMMAND [ARGS...]
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/spur.env"
mkdir -p "$DOCKER_CONFIG"
opts=(-u "$JT_USER" -w "$JT_REPO" -e USER="$JT_USER" -e TRAIN_TIMEOUT)
if [[ "${1:-}" == -d ]]; then
  opts+=(-d)
  shift
fi
exec docker exec "${opts[@]}" "$JT_CONTAINER" "$@"
EOF
chmod +x "$JT_SPUR/container.sh" "$JT_SPUR/jt.sh"
```

### Scripts that run in the container

```bash
cat > "$JT_SPUR/build.sh" <<'EOF'
#!/usr/bin/env bash
# Builds the JAX wheels into $JT_ROOT/src/jax/dist, with the Bazel output root
# on node-local scratch. Arguments are extra settings such as WHEELS=jaxlib.
# Logs to $JT_ROOT/logs/build_jax_rocm.log.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/spur.env"
exec > "$JT_ROOT/logs/build_jax_rocm.log" 2>&1
trap 'echo "$(date -Is) exit=$?"' EXIT
env BAZEL_ROOT="$JT_SCRATCH/bazel" "$@" scripts/build_jax_rocm.sh
EOF

cat > "$JT_SPUR/venv.sh" <<'EOF'
#!/usr/bin/env bash
# Creates the venv as in getting_started.md section 4, plus orbax-checkpoint and
# pytest, which pyproject.toml lists.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/spur.env"
JAX_TITAN_ROOT=$JT_ROOT
python3 -m venv --system-site-packages ${JAX_TITAN_ROOT}/venv
source ${JAX_TITAN_ROOT}/venv/bin/activate
D=${JAX_TITAN_ROOT}/src/jax/dist
pip install $D/jax-*.whl $D/jaxlib-*.whl
pip install --no-deps $D/jax_rocm10_plugin-*.whl $D/jax_rocm10_pjrt-*.whl
pip install optax tiktoken pyarrow jaxtyping absl-py immutabledict typing_extensions chex
pip install --no-deps -e ${JAX_TITAN_ROOT}/src/shardy/shardy/integrations/python/jax/mpmd
pip install orbax-checkpoint pytest
EOF

cat > "$JT_SPUR/run.sh" <<'EOF'
#!/usr/bin/env bash
# Runs a command after scripts/env.sh.
set -euo pipefail
source scripts/env.sh
exec "$@"
EOF

cat > "$JT_SPUR/train.sh" <<'EOF'
#!/usr/bin/env bash
# Runs the trainer with XLA's ROCm flags, as in getting_started.md. Appends its
# output to $JAX_TITAN_ROOT/logs/NAME.out and per-step metrics to NAME.jsonl;
# NAME may include a directory, such as spur/zb_pp2. With TRAIN_TIMEOUT set
# (for example 30m), a run that takes longer is killed.
#   train.sh NAME [TRAINER FLAGS...]
set -euo pipefail
source scripts/env.sh
name=$1
shift
log=$JAX_TITAN_ROOT/logs/$name
mkdir -p "$(dirname "$log")"
exec >> "$log.out" 2>&1
trap 'echo "$(date -Is) exit=$?"' EXIT
echo "$(date -Is) $(hostname) train.sh $name $*"
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" ${TRAIN_TIMEOUT:+timeout -s KILL "$TRAIN_TIMEOUT"} \
  python -m jax_titan.trainer.train --log_file="$log.jsonl" "$@"
EOF
```

`build.sh` and `train.sh` end their logs with a line like
`2026-11-05T21:23:56+00:00 exit=0`, so you can tell from the login node when
a detached run has finished and how.

## 3. Get a node

If you were given a JobID, use it and skip the `sbatch` below; agents only
allocate when asked (rule 2 in spur-cluster-guide.md). To hold a node for
building, testing and several runs (hold mode, tested on v2):

```bash
cd /tmp
sbatch --controller "$SPUR" --parsable \
  -J "$USER-hold-jax_titan" -p "$PART" \
  -N1 -w "$NODE" --gpus-per-node=8 --exclusive \
  -t 1-00:00:00 --wrap "sleep 86400"
squeue --controller "$SPUR" -u "$USER"         # wait for ST=R
```

Then, in every new login shell:

```bash
JOBID=<JobID>
JT_SPUR=$HOME/jax_titan/.jax_titan/spur          # as in section 2
source "$JT_SPUR/spur.env"
spur show job --controller "$SPUR" "$JOBID" | head -20    # RUNNING, TresPerNode=gpu:8/node
```

A hold-mode job keeps the node until you cancel it (section 16) or its time
limit runs out; renew it as in spur-cluster-guide.md section 7. For a single
training run that releases the node by itself, see section 15.

## 4. Check the node

```bash
on_node 'hostname; nproc; uptime; free -h; rocm-smi'
on_node 'whoami; id; echo "HOME=$HOME PWD=$PWD"; ls -d "$JT_REPO"; docker ps -a'
on_node 'df -h "$SCRATCH"'
gpus_free
```

- `rocm-smi` lists 8 GPUs.
- `whoami` and `id` show who `spur exec` runs as. The source saw root on the
  old pool and a `HOME` that is not yours on v2. Nothing below depends on it:
  container.sh uses the identity recorded in spur.env, and everything that
  writes to the shared home runs in the container as your uid. If `docker ps`
  fails with a permission error, stop and ask; do not work around it.
- `ls` shows the checkout, so the shared home is mounted on the node.
- `docker ps -a` lists the node's containers. Leftover containers, yours or an
  earlier holder's, can still hold GPU memory.
- `gpus_free.sh` must print `free` (section 11).

## 5. Start the container

Pull the image in the foreground. Background processes started from `srun`
steps were killed when the step ended (spur-cluster-guide.md section 5):

```bash
on_node 'docker pull "$JT_IMAGE"'
on_node 'docker image inspect --format "{{index .RepoDigests 0}}" "$JT_IMAGE"'
```

The venv uses the image's Python and ROCm libraries, so all nodes should run
the same image. After the first pull, consider setting `JT_IMAGE` in spur.env
to the printed `ghcr.io/...@sha256:...` digest instead of the `latest` tag.

Start the container and check that it runs as you:

```bash
on_node 'bash $JT_SPUR/container.sh'
in_jt id                       # your uid, gid, and the video and render groups
in_jt ls -l /dev/kfd /dev/dri
```

The image brings its own ROCm user space, TheRock 10.1; only the kernel
driver comes from the node. The RL runbooks ran images whose ROCm differed
from the host's without trouble, but nobody has run TheRock 10.1 on a SPUR
node. The quick GPU check in section 10 is the first test of that.

What differs from `scripts/container.sh` and getting_started.md section 1:

- You start the container through `spur exec`, on the node's own docker
  daemon. There is no ssh and no terminal, so `docker exec -it` does not
  work: jt.sh calls `docker exec` without `-it`, and with `-d` for anything
  long.
- `spur exec` may not run as you, so the user, uid and gid come from spur.env,
  recorded on the login node, instead of `$USER` and `id`.
- docker needs a writable configuration directory (`DOCKER_CONFIG`).
- The mounted workspace, the checkout's parent, is on the NFS shared home. The
  container user has your uid, so files there keep your owner.
- container.sh also mounts `$JT_SCRATCH` (node-local) for the Bazel output root
  and throwaway checkpoints, and gives it to your uid.
- docker's root directory is on node-local disk, so every new node pulls the
  image again and needs a new container. Wheels, venv and data on the shared
  home carry over.
- container.sh reuses an existing container instead of failing on the name.
- `scripts/gpus_free.sh` runs outside the container (`on_node`), where it sees
  every user's processes.

## 6. Build the wheels, or reuse them

The wheels land in `$JT_ROOT/src/jax/dist` on the shared home, so you build
them once and reuse them on any node. If the four wheels are there
(`jax`, `jaxlib`, `jax_rocm10_plugin`, `jax_rocm10_pjrt`), skip this section:

```bash
ls -l "$JT_ROOT/src/jax/dist"        # on the login node
```

Otherwise build them, detached, and follow the log from the login node:

```bash
in_jt -d bash $JT_SPUR/build.sh
tail -f "$JT_ROOT/logs/build_jax_rocm.log"     # done at the exit= line; then ls dist
```

getting_started.md gives about 25 minutes on 256 cores; the build has not been
timed on SPUR. It downloads Bazel dependencies and the TheRock tarball, so the
node needs internet access; nodes on the old pool pulled images and downloaded
models in the RL runbooks. Only build on one node at a time, since every node
writes the same `dist` directory.

- jaxlib only: `in_jt -d bash $JT_SPUR/build.sh WHEELS=jaxlib`.
- `build_jax_rocm.sh` caps Bazel at 160 CPUs (`BAZEL_CPUS`) to leave headroom
  on shared machines. On an exclusive node with 236 CPUs you can pass, for
  example, `BAZEL_CPUS=220`.
- A rebuilt wheel keeps its version string, so reinstall it with
  `pip install --force-reinstall --no-deps` to be sure it replaces the
  installed one.

## 7. Create the venv

Skip this if `$JT_ROOT/venv` exists. Otherwise:

```bash
in_jt bash $JT_SPUR/venv.sh
in_jt bash $JT_SPUR/run.sh pip list | grep -iE '^(jax|optax|orbax|shardy|pytest)'
```

venv.sh runs the commands of getting_started.md section 4 and also installs
`orbax-checkpoint` (needed by `--checkpoint_dir` and the checkpoint tests) and
`pytest`. Both are in `pyproject.toml` but not in getting_started.md's install
line; the venv JAX-Titan was tested in has them. From here on, every command
in the container sources `scripts/env.sh` first (run.sh and train.sh do).

## 8. Prepare the data

```bash
in_jt bash $JT_SPUR/run.sh python -m jax_titan.data.text
```

This runs on the CPU. It downloads WikiText-103 from Hugging Face into
`$JT_ROOT/data` and writes `wikitext103_train.llama3.bin` and
`wikitext103_validation.llama3.bin`, printing the token count of each; the
training file has about 120.1M tokens. It needs internet access and the
MaxText checkout next to JAX-Titan.

## 9. Run the CPU tests

```bash
in_jt bash $JT_SPUR/run.sh python -m pytest tests/
```

The tests compare every schedule and layout with `jax.jit` and check
checkpoint resume. They do not use the GPUs: `tests/conftest.py` sets
`JAX_PLATFORMS=cpu` with 8 CPU devices.

## 10. Train

Before **every** GPU run, check on the host that the GPUs are free
(section 11):

```bash
gpus_free
```

Optionally, a quick check that MPMD runs on this node's GPUs, from
getting_started.md (a two-stage MLP step on two 4-GPU meshes, checked against
`jax.jit`):

```bash
gpus_free && \
in_jt bash $JT_SPUR/run.sh python examples/mpmd_two_mesh_spike.py --platform=rocm
```

### Zero-bubble pipeline

getting_started.md trains Llama 3.1 8B with 2 stages of 4 MI355X and the
zero-bubble schedule (`ZERO_BUBBLE_H1`, the trainer's default) like this:

```bash
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=llama3.1-8b --num_stages=2 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100
```

train.sh runs exactly that with the flags you pass, plus `--log_file`. On
SPUR, detached:

```bash
gpus_free && \
in_jt -d bash $JT_SPUR/train.sh zb_pp2 \
  --model=llama3.1-8b --num_stages=2 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100
```

### No-pipeline baseline

The same model, weights and batches as one `jax.jit` program on all 8 GPUs,
with parameters and optimizer state sharded (`--parallel=fsdp`; `--parallel=dp`
replicates them):

```bash
gpus_free && \
in_jt -d bash $JT_SPUR/train.sh fsdp8 \
  --model=llama3.1-8b --parallel=fsdp \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100
```

pipelines.md describes `--schedule`, `--stage_fsdp`, `--tensor_parallel`,
`--stage_layers` and `--reduce_grads`.

### Follow a run

The logs are on the shared home, so follow them from the login node:

```bash
tail -f "$JT_ROOT/logs/zb_pp2.out"
until tail -n 1 "$JT_ROOT/logs/zb_pp2.out" | grep -q 'exit='; do sleep 30; done
on_node 'rocm-smi --showuse'        # busy GPUs while it trains
```

`in_jt -d` returns at once. **Run one GPU job at a time:** wait for the
`exit=` line, and check `gpus_free.sh` again, before starting the next run.

A healthy start prints these lines, in this order (from a run on the
development node, with the measured values replaced):

```text
jax <version> | 8 x AMD Instinct MI355X | llama3.1-8b (8.03B params) | 2 stages | schedule=ZERO_BUBBLE_H1 | gradients reduced per step
initialized params and optimizer state in <s>s
lowered in <s>s, compiled in <s>s
data: <JAX_TITAN_ROOT>/data/wikitext103_train.llama3.bin (120.1M tokens)
step    0 | loss <loss> | <ms> ms | <tokens/s> tok/s | MFU <percent>%
peak HBM per device (GiB): <one value per GPU>
```

- Before these lines, `GetPjrtApi was found for rocm` shows that the ROCm
  plugin loaded, and the ROCm runtime prints many `queue-intercept` warnings;
  those appeared in successful runs too.
- Pipeline runs log each mesh's fragment order in lines starting
  `Module jit_train_step on mesh` while compiling (pipelines.md).
- Step 0 takes much longer than the steps after it.
- Every run starts from the same weights, so the step-0 loss should match the
  one given with the results in pipelines.md.
- The trainer stops with `FloatingPointError` if the loss is not finite.

Give every run a time limit, sized from an earlier run, as the repository's
training skill (`.cursor/skills/jax_titan_training`) asks; a hung run does not
end by itself (section 11). Untested on SPUR:

```bash
gpus_free && \
TRAIN_TIMEOUT=30m in_jt -d bash $JT_SPUR/train.sh zb_pp2 \
  --model=llama3.1-8b --num_stages=2 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100
```

To stop a run, stop the process in the container. Interrupting `spur exec`
or the `docker exec` client does not stop it:

```bash
on_node 'docker exec "$JT_CONTAINER" pkill -KILL -f "[j]ax_titan.trainer.train"'
```

`scripts/cleanup_after_train.sh` (section 16) does the same and more.

## 11. Lessons from a shared node

So far, JAX-Titan's GPU runs happened on a shared 8x MI355X node outside
SPUR, where other users' jobs could start at any time.

- **Check `scripts/gpus_free.sh` on the host before every GPU run.** The host
  is outside the JAX-Titan container, where the script sees every user's
  processes; on SPUR that is `on_node`, not `in_jt`. It prints `free` only
  when no process holds a GPU and no GPU is above 5% use or 5% VRAM
  (`MAX_USE_PCT`, `MAX_VRAM_PCT`). Otherwise it prints why and exits 1.
- **Another job on the same GPUs hung the run in RCCL setup.** In the
  incident, the node was idle when the run started, and another user's job
  then started on the same GPUs. The JAX-Titan run hung in RCCL communicator
  setup at step 0: XLA logged that the rendezvous leader was stuck in
  `Initialize clique`, and nothing progressed. Two attempts of a 30-step run
  were killed by a 30-minute timeout while the other user's processes were on
  the GPUs; only the rerun on the idle node completed, in about 3.5 minutes.
  Runs therefore need a time limit (`TRAIN_TIMEOUT`, section 10). The signs:
  no `step    0` line within about 3 minutes of `compiled in` (the threshold
  the repository's training skill uses), and
  `grep -nE "may be stuck|Initialize clique" <log>` finds XLA's rendezvous
  warning. Stop the run, wait until `gpus_free.sh` prints `free` again, and
  rerun.
- **JAX preallocates 75% of each GPU's memory.** Two jobs on the same GPUs
  leave neither enough room, and the 8B runs use most of their pool (see the
  peak HBM per GPU in pipelines.md). This includes your own runs: never start
  a second one while the first is running.
- **The memory takes a moment to come back.** On the shared node,
  `gpus_free.sh` run right after a run had exited reported `vram 75%` on all
  eight GPUs; the next check, 30 seconds later, passed.

On SPUR, an exclusive job should keep other users' jobs off your node's GPUs,
but containers outlive jobs: a finished task once left about 90.9 GB in use on
every GPU for the next user (spur-cluster-guide.md section 8). If
`gpus_free.sh` reports busy, look at `on_node 'rocm-smi --showpids; docker ps -a'`.

## 12. Checkpoints

`--checkpoint_dir=DIR` saves parameters and optimizer state with Orbax after
the last step and every `--checkpoint_every` steps. Rerunning the same command
resumes from the latest checkpoint in `DIR`, including the data position. The
getting_started.md example, with FSDP inside each stage and 17 and 15 layers
per stage:

```bash
gpus_free && \
in_jt -d bash $JT_SPUR/train.sh zb_fsdp_ckpt \
  --model=llama3.1-8b --num_stages=2 --stage_fsdp --stage_layers=17,15 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100 \
  --checkpoint_dir=$JT_ROOT/checkpoints/llama8b --checkpoint_every=50
```

Each save prints `saved step <N> in <s>s`. A rerun prints
`restored step <N> from <DIR> in <s>s` instead of `initialized ...`; since
train.sh appends, the resumed run continues the same `.out` and `.jsonl`
files.

- **Keep checkpoints you need on the shared home.** Jobs can end early without
  warning, node-local scratch does not move with you and has been wiped
  (spur-cluster-guide.md sections 9 and 10). `$JT_ROOT/checkpoints` is on the
  shared home.
- **Budget the space.** A Llama 3.1 8B checkpoint is 80 GB and the trainer
  keeps the latest two in each directory: about 160 GB per directory, more
  while a new one is being written.
- **Saving blocks training.** Saving blocked training for 31-39 s while the
  arrays were copied to host memory, and Orbax then wrote the checkpoint in
  the background (pipelines.md). Checkpoint less often than every few minutes
  until saving is faster. How long the background write to the shared home
  takes on SPUR is untested.
- **Throwaway checkpoints**, for example to test resuming, can go to
  `--checkpoint_dir=$JT_SCRATCH/checkpoints/<name>` (node-local, mounted in
  the container) or to `/dev/shm` (memory, node-local, gone on reboot). Never
  keep the only copy of a checkpoint you need there.
- **One model and structure per directory.** The trainer refuses a checkpoint
  of another model or structure (`mpmd:<stages>` for pipelines, `spmd` for
  `--parallel=dp` and `fsdp`) with a `ValueError`. It does restore into another
  layout of the same stages, for example with `--stage_fsdp` added.
- The files belong to you, since the container runs as your uid. Delete
  checkpoints you no longer need from the login node, or with
  `scripts/cleanup_after_train.sh --checkpoint DIR` (section 16), which only
  removes Orbax checkpoints below `/dev/shm`, `/tmp` or `$JT_ROOT`. Ask
  before deleting a checkpoint that a run may resume from.

## 13. Profiling

`--profile_steps=START:END` traces steps `[START, END)` with `jax.profiler`
into `--profile_dir`, by default `$JT_ROOT/profiles` on the shared home. Give
each run its own directory, because the summary reads the newest trace under
the directory you name. Start after the first steps, since step 0 is slow.

```bash
gpus_free && \
in_jt -d bash $JT_SPUR/train.sh zb_pp2_profile \
  --model=llama3.1-8b --num_stages=2 \
  --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=20 \
  --profile_steps=10:12 --profile_dir=$JT_ROOT/profiles/zb_pp2
```

The log prints `traced steps 10-11 into <dir>`. Then summarize the trace, on
the CPU:

```bash
in_jt bash $JT_SPUR/run.sh python -m jax_titan.observability.profile $JT_ROOT/profiles/zb_pp2
```

It prints, for each GPU, the share of the traced window spent in compute
kernels, collectives and memory copies, and idle. For a pipeline stage, idle
is its bubble plus any wait for transfers. `--lines` lists the trace's planes
and lines instead. Leave the traced steps out of step-time medians: the first
one carries the profiler's startup cost. Trace sizes and tracing to the shared
home are untested on SPUR; remove traces you no longer need.

## 14. Logs

Everything is on the shared home and readable from the login node.

| File | Written by | Contents |
|---|---|---|
| `$JT_ROOT/logs/<name>.out` | train.sh | Trainer, XLA and Shardy output; a dated first line and an `exit=` line per run |
| `$JT_ROOT/logs/<name>.jsonl` | `--log_file` | One JSON object per step: `step`, `loss`, `step_time_s`, `tokens_per_s`, `mfu` |
| `$JT_ROOT/logs/build_jax_rocm.log` | build.sh | The Bazel build and an `exit=` line |
| `$JT_ROOT/logs/<JobID>.out`, `.err` | `sbatch` | Task-mode job scripts (section 15) |

Both train.sh files are appended to, so use a new name for each new run, and
the same name only to resume. Compare `.jsonl` files with the reference runs
in `docs/results/`. To tell a hang from slow progress, the RL runbooks looked
at three things: whether the log still changes, whether the process is alive,
and GPU use (`on_node 'rocm-smi --showuse'`).

## 15. One training run in task mode

**Untested.** Task mode ends the job, and releases the node, when the job
script exits (spur-cluster-guide.md section 4.1). This script checks the
GPUs, starts the container and runs one command in it. When it exits, it
runs `scripts/cleanup_after_train.sh --yes` in the container and removes the
container, so the next user finds the GPUs free. Whether docker works
from a task-mode job script, and whether `-c` limits the container (docker
starts it outside the job's cgroup), has not been checked.

```bash
cat > "$JT_SPUR/task.sh" <<'EOF'
#!/usr/bin/env bash
# Task-mode job script: checks that the GPUs are free, starts the container,
# runs one command in it, and when the script exits cleans up after the run
# and removes the container.
#   task.sh COMMAND [ARGS...]
set -euo pipefail
d=$(dirname "$(readlink -f "$0")")
source "$d/spur.env"
echo "$(date -Is) $(hostname) nproc=$(nproc)"
"$JT_REPO/scripts/gpus_free.sh"
trap '"$d/jt.sh" scripts/cleanup_after_train.sh --yes || true
  docker rm -f "$JT_CONTAINER" >/dev/null 2>&1 || true' EXIT
bash "$d/container.sh"
"$d/jt.sh" "$@"
EOF
```

Run it through `--wrap`, so that it runs from the shared home: `sbatch`
copies a submitted script file to the node, where `$0` no longer points next
to spur.env. Pass `-c` and `--exclusive`, and keep the logs on the shared
home:

```bash
cd /tmp
sbatch --controller "$SPUR" --parsable -J "$USER-jax_titan" -p "$PART" \
  -N1 -w "$NODE" --gpus-per-node=8 --exclusive -c 64 -t 04:00:00 \
  -o "$JT_ROOT/logs/%j.out" -e "$JT_ROOT/logs/%j.err" \
  --wrap "bash $JT_SPUR/task.sh bash $JT_SPUR/train.sh zb_pp2_task \
    --model=llama3.1-8b --num_stages=2 \
    --microbatches=8 --microbatch_size=4 --seq_len=2048 --steps=100"
```

Confirm `R` with `squeue`; afterwards, `sacct --controller "$SPUR" | grep <JobID>`
shows `COMPLETED` and `0:0` on success. The trainer's output is in
`$JT_ROOT/logs/zb_pp2_task.out`, the job script's in `<JobID>.out`. The trap
does not run if the job script is killed outright, and once the job is gone,
`spur exec` cannot reach the node; a container left behind then holds the
GPUs for the next user.

## 16. Clean up and release the node

As soon as a run finishes, fails, hangs or is no longer needed, free the GPUs
with `scripts/cleanup_after_train.sh`. It stops your JAX-Titan processes,
waits until none of your processes hold a GPU, removes your stale RCCL
segments in `/dev/shm`, and removes the checkpoint directories you name; it
never touches other users' processes or files. getting_started.md runs it on
the host. On SPUR, run it in the container, where it runs as you; `spur exec`
may run as someone else, and the script acts only on `$USER`'s processes and
files. Untested on SPUR:

```bash
in_jt scripts/cleanup_after_train.sh                     # dry run: prints the plan
in_jt scripts/cleanup_after_train.sh --yes               # apply
in_jt scripts/cleanup_after_train.sh --yes --checkpoint $JT_ROOT/checkpoints/<name>
gpus_free
```

It removes checkpoints only below `/dev/shm`, `/tmp` or `$JT_ROOT`, so it
skips throwaway checkpoints under `$JT_SCRATCH`; remove those as described
below. Also remove profiler traces and other large outputs you no longer
need, and keep the logs.

In hold mode, then remove the container, check that the GPUs are free, and
cancel the job:

```bash
on_node 'docker rm -f "$JT_CONTAINER"'
gpus_free                                   # idle memory is about 298 MB per GPU
scancel --controller "$SPUR" "$JOBID"
squeue --controller "$SPUR" -u "$USER"      # the job is gone
```

Only cancel a job you allocated or whose owner asked you to. Remove containers
before the job ends: afterwards, `spur exec` cannot reach the node.

The shared home keeps the wheels, venv, data, logs and checkpoints for the next
node. Node-local scratch can stay; it belongs to this node and may be wiped
anyway. To clean it, do so before removing the container, as root in the
container, because Bazel leaves read-only directories:

```bash
on_node 'docker exec "$JT_CONTAINER" find "$JT_SCRATCH" -mindepth 1 -delete'
```

## 17. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `the input device is not a TTY` | `docker exec -it` through `spur exec` | Use `in_jt` or jt.sh, which do not pass `-it` |
| container.sh prints `no video group` or `no render group` | The `spur exec` environment lacks the group | In container.sh, set `VIDEO_GID` and `RENDER_GID` to the groups of `/dev/dri/card*` and `/dev/kfd` (`stat -c %g`) |
| `useradd` or `groupadd` errors, or `in_jt id` does not show your uid | spur.env written by someone else, or the gid already exists in the image | Rewrite spur.env on the login node as yourself; `docker rm -f` the container and start it again |
| `docker` reports a permission error on the node | No docker access from `spur exec` | Stop and ask |
| `venv/bin/activate: No such file or directory` | The venv does not exist yet | Section 7 |
| `No module named 'orbax'` or `'pytest'` | venv made with getting_started.md's install line only | `pip install orbax-checkpoint pytest` (venv.sh does) |
| `tokenizer_llama3.tiktoken` not found | No MaxText checkout next to JAX-Titan | Clone it (section 2), or `in_jt bash $JT_SPUR/run.sh env JAX_TITAN_LLAMA3_TOKENIZER=<file> python -m jax_titan.data.text` |
| No `step    0` within about 3 minutes of `compiled in`; `may be stuck` and `Initialize clique` in the log | Another job on the GPUs during RCCL setup | Stop the run; wait for `free`; rerun (section 11) |
| The run keeps going after you interrupted `spur exec` or `docker exec` | The process lives in the container | Stop it in the container (section 10) |
| `/dev/shm` collects `nccl-*` files | Killed runs leave RCCL segments behind | `scripts/cleanup_after_train.sh --yes` (section 16) |
| Out of GPU memory at the start of a run | Another process holds GPU memory | `gpus_free.sh`; remove leftover containers |
| `gpus_free.sh` reports `vram 75%` right after your run | The memory is still being released | Check again 30 seconds later |
| `ValueError: checkpoint holds ... not ...` | `--checkpoint_dir` holds another model or structure | One directory per model and structure |
| The shared home fills up | Checkpoints (80 GB each, two kept per directory) or GPU core dumps | Delete old checkpoints and `gpucore.*` files; throwaway checkpoints to scratch |
| `Memory access fault by GPU node-<n>`, then `gpucore.<pid>` files of 50-80 GB | ROCm writes a GPU core dump into the working directory; seen with PyTorch on ROCm 7.0, untested with JAX-Titan | The working directory is the checkout on the shared home: delete `$JT_REPO/gpucore.*` at once. `HSA_ENABLE_COREDUMP=0` reportedly disables them; untested |
| The job ended with a container still running | Containers outlive jobs | Remove containers before the job ends (section 16) |

## 18. Checklist

- [ ] Read the rules and Cluster facts in spur-cluster-guide.md.
- [ ] Checkouts and `$JT_ROOT/src` on the shared home; spur.env and the
      scripts written on the login node as yourself.
- [ ] The job is `RUNNING`; `JOBID` is set and spur.env is sourced.
- [ ] `rocm-smi` shows 8 GPUs; `gpus_free.sh` prints `free`.
- [ ] Container started; `in_jt id` shows your uid.
- [ ] Wheels in `$JT_ROOT/src/jax/dist` and the venv in `$JT_ROOT/venv`.
- [ ] Data in `$JT_ROOT/data`; CPU tests pass.
- [ ] Before each GPU run, `gpus_free.sh` prints `free` and none of your runs
      is still going.
- [ ] Each new run has its own name; you waited for `exit=` before the next.
- [ ] Checkpoints you keep are on the shared home, and it has room.
- [ ] After each run: `scripts/cleanup_after_train.sh --yes` in the
      container, and `gpus_free.sh` prints `free`.
- [ ] At the end: container removed, `gpus_free.sh` prints `free`, and the
      job cancelled (hold mode) or `COMPLETED` with `0:0` (task mode).
