#!/usr/bin/env bash
# Starts the JAX-Titan development container (TheRock 10.1) with all GPUs and
# adds a user matching the host uid, so files written to the workspace keep the
# host owner. Run on the host.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
NAME=${NAME:-${USER}_therock-10.1_jaxtitan}
IMAGE=${IMAGE:-ghcr.io/rocm/jax-base-ubu24.therock-10.1:latest}
# Mount the repo's parent so sibling checkouts (maxtext) and .jaxtitan are visible.
WORKSPACE=${WORKSPACE:-$(dirname "${REPO}")}
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

docker run -d --name "${NAME}" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "${VIDEO_GID}" --group-add "${RENDER_GID}" \
  --ipc=host --shm-size=64G --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE --network=host \
  -v "${WORKSPACE}:${WORKSPACE}" -w "${REPO}" \
  "${IMAGE}" sleep infinity

docker exec "${NAME}" bash -lc "
  getent group ${VIDEO_GID} || groupadd -g ${VIDEO_GID} video
  getent group ${RENDER_GID} || groupadd -g ${RENDER_GID} render
  groupadd -g $(id -g) $(id -gn)
  useradd -u $(id -u) -g $(id -g) -G ${VIDEO_GID},${RENDER_GID} \
    -d /home/${USER} -M -s /bin/bash ${USER}
  mkdir -p /home/${USER} && chown $(id -u):$(id -g) /home/${USER}"

echo "docker exec -it -u ${USER} ${NAME} bash"
