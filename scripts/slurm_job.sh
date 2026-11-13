#!/usr/bin/env bash
# Slurm batch script for one node: checks the node, starts the TheRock
# container, builds the JAX wheels and the venv when they are missing, then
# runs the trainer in the container. Each setup stage is skipped when its
# output exists, so a resubmitted job continues where the last one stopped. On
# exit, also after SIGTERM from scancel or the time limit, it runs
# scripts/cleanup_after_train.sh in the container and removes the container,
# so the next user finds the GPUs free.
#
# Submit it with scripts/slurm_submit.sh, which runs it through --wrap from the
# shared filesystem (sbatch copies a submitted script file to the node, where
# it would no longer sit next to the repository).
#
#   slurm_job.sh [--setup-only] [TRAINER FLAGS...]
#
# Needs: docker usable by the job user, AMD GPUs (/dev/kfd), the repository and
# JAX_TITAN_ROOT on a filesystem shared with the login node, and internet
# access for the first build. Environment (all optional):
#   JT_SITE_ENV     a file of these settings for your cluster, sourced first
#   JAX_TITAN_ROOT  build trees, wheels, venv and logs (default: .jax_titan next to the repo)
#   JT_IMAGE        container image (default: ghcr.io/rocm/jax-base-ubu24.therock-10.1:latest)
#   JT_SCRATCH      node-local directory for the Bazel output root
#                   (default: $SLURM_TMPDIR, else $TMPDIR, else /tmp; plus /$USER/jax_titan)
#   TRAIN_TIMEOUT   time limit of the trainer, for timeout(1) (default: 45m)
#   BAZEL_CPUS      Bazel CPU cap (default: nproc - 16)
set -euo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
WORKSPACE=$(dirname "$REPO")
if [[ -n ${JT_SITE_ENV:-} ]]; then
  # shellcheck source=/dev/null
  source "$JT_SITE_ENV"
fi
export JAX_TITAN_ROOT=${JAX_TITAN_ROOT:-$WORKSPACE/.jax_titan}
IMAGE=${JT_IMAGE:-ghcr.io/rocm/jax-base-ubu24.therock-10.1:latest}
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-45m}
JOB=${SLURM_JOB_ID:-manual-$$}

SETUP_ONLY=false
if [[ ${1:-} == --setup-only ]]; then
  SETUP_ONLY=true
  shift
fi

U=$(id -un) UID_=$(id -u) GID_=$(id -g) GNAME=$(id -gn)
CNAME=${U}_therock-10.1_jax_titan_${JOB}
JT_SCRATCH=${JT_SCRATCH:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/$U/jax_titan}
# docker needs a writable configuration directory; HOME may not be writable.
export DOCKER_CONFIG=${DOCKER_CONFIG:-/tmp/.docker-$UID_}
mkdir -p "$DOCKER_CONFIG" "$JT_SCRATCH" "$JAX_TITAN_ROOT/logs/slurm"

stage() { echo; echo "=== $(date -Is) $*"; }
die() { local rc=$1; shift; echo "error: $*" >&2; echo "error: $*"; exit "$rc"; }

stage "node"
echo "host=$(hostname) user=$U uid=$UID_ job=$JOB nproc=$(nproc) repo=$REPO root=$JAX_TITAN_ROOT scratch=$JT_SCRATCH"
echo "branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null) commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
df -h "$JT_SCRATCH" | tail -1
[[ -e /dev/kfd ]] || die 7 "no /dev/kfd on $(hostname): not an AMD GPU node"
command -v docker >/dev/null || die 7 "docker is not installed on $(hostname)"
docker info >/dev/null 2>&1 || die 7 "docker is not usable by $U on $(hostname) (daemon down or no permission); this script needs docker"
echo "docker $(docker version --format '{{.Server.Version}}')"
for d in jax xla shardy; do
  [[ -d $JAX_TITAN_ROOT/src/$d ]] || die 2 "missing $JAX_TITAN_ROOT/src/$d: run scripts/setup_sources.sh on the login node"
done
"$REPO/scripts/gpus_free.sh" || die 3 "GPUs are busy"

STARTED=false
cleanup() {
  local rc=$?
  trap - EXIT TERM INT
  stage "cleanup (exit=$rc)"
  if $STARTED; then
    docker exec -u "$UID_" -e USER="$U" -w "$REPO" "$CNAME" \
      scripts/cleanup_after_train.sh --yes 2>&1 | tail -6 || true
    docker rm -f "$CNAME" >/dev/null 2>&1 || true
  fi
  # The container's processes can hold a GPU for a few seconds after docker rm.
  local i
  for i in $(seq 12); do
    "$REPO/scripts/gpus_free.sh" >/dev/null 2>&1 && break
    sleep 5
  done
  echo "GPU state after $((i * 5 - 5)) s: $("$REPO/scripts/gpus_free.sh" 2>&1 | head -3 | tr '\n' ' ')"
  echo "$(date -Is) job exit=$rc"
}
trap cleanup EXIT
# scancel and the time limit send SIGTERM; exit so that the EXIT trap cleans up.
trap 'echo "$(date -Is) received SIGTERM/SIGINT"; exit 143' TERM INT

# Runs a command in the background and waits for it, so that a signal reaches
# the traps at once instead of after the command ends.
run() { "$@" & wait $!; }

stage "container"
run docker pull -q "$IMAGE"
docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" || true
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)
docker rm -f "$CNAME" >/dev/null 2>&1 || true
STARTED=true
docker run -d --name "$CNAME" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
  --ipc=host --shm-size=64G --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE --network=host \
  -v "$WORKSPACE:$WORKSPACE" -v "$JT_SCRATCH:$JT_SCRATCH" -w "$REPO" \
  "$IMAGE" sleep infinity >/dev/null
# The container user gets the job user's uid and gid, so files keep their owner.
docker exec "$CNAME" bash -c "
  getent group $VIDEO_GID >/dev/null || groupadd -g $VIDEO_GID video
  getent group $RENDER_GID >/dev/null || groupadd -g $RENDER_GID render
  getent group $GID_ >/dev/null || groupadd -g $GID_ $GNAME
  getent passwd $UID_ >/dev/null || useradd -u $UID_ -g $GID_ -G $VIDEO_GID,$RENDER_GID -d /home/$U -M -s /bin/bash $U
  mkdir -p /home/$U && chown $UID_:$GID_ /home/$U $JT_SCRATCH"

# Runs a bash command string in the container as the job user.
in_ct() {
  docker exec -u "$UID_" -e USER="$U" -e HOME="/home/$U" -e JAX_TITAN_ROOT="$JAX_TITAN_ROOT" \
    -w "$REPO" "$CNAME" bash -c "$1"
}
in_ct 'id; python3 --version'
in_ct "test -r $REPO/scripts/env.sh" || die 7 "the container cannot read $REPO: is it on a filesystem the node mounts?"

DIST=$JAX_TITAN_ROOT/src/jax/dist
if ls "$DIST"/jaxlib-*.whl "$DIST"/jax_rocm10_plugin-*.whl "$DIST"/jax_rocm10_pjrt-*.whl >/dev/null 2>&1; then
  stage "build: wheels exist in $DIST, skipping"
else
  BUILD_LOG=$JAX_TITAN_ROOT/logs/build_jax_rocm.log
  stage "build wheels (log: $BUILD_LOG)"
  run in_ct "BAZEL_ROOT=$JT_SCRATCH/bazel BAZEL_CPUS=${BAZEL_CPUS:-\$((\$(nproc) - 16))} scripts/build_jax_rocm.sh" \
    > "$BUILD_LOG" 2>&1 || { tail -40 "$BUILD_LOG"; die 4 "wheel build failed"; }
  tail -6 "$BUILD_LOG"
fi

VENV=$JAX_TITAN_ROOT/venv
if [[ -f $VENV/.jax_titan_ok ]]; then
  stage "venv: $VENV exists, skipping"
else
  stage "venv"
  rm -rf "$VENV"
  run in_ct "set -e
    python3 -m venv --system-site-packages $VENV
    source $VENV/bin/activate
    pip install -q $DIST/jax-*.whl $DIST/jaxlib-*.whl
    pip install -q --no-deps $DIST/jax_rocm10_plugin-*.whl $DIST/jax_rocm10_pjrt-*.whl
    pip install -q optax tiktoken pyarrow jaxtyping absl-py immutabledict typing_extensions chex orbax-checkpoint pytest
    pip install -q --no-deps -e $JAX_TITAN_ROOT/src/shardy/shardy/integrations/python/jax/mpmd
    touch $VENV/.jax_titan_ok" || die 5 "venv setup failed"
fi

stage "gpu check"
in_ct "source scripts/env.sh && python -c 'import jax; print(jax.__version__, jax.devices())'" 2>&1 \
  | grep -v queue-intercept || die 6 "JAX sees no GPUs"

if $SETUP_ONLY; then
  stage "done (setup only)"
  exit 0
fi

LOG=$JAX_TITAN_ROOT/logs/slurm/$JOB
stage "train (timeout $TRAIN_TIMEOUT, step records: $LOG.jsonl): $*"
# printf %q keeps each trainer flag one word through bash -c.
run in_ct "source scripts/env.sh && export JAX_TITAN_LAUNCH_TIME=\$(date +%s.%N) && \
  XLA_FLAGS=\"\$JAX_TITAN_ROCM_XLA_FLAGS\" timeout -s KILL $TRAIN_TIMEOUT \
  python -m jax_titan.trainer.train --log_file=$LOG.jsonl $(printf '%q ' "$@")"
stage "done"
