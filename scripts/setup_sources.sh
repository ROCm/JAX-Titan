#!/usr/bin/env bash
# Creates the jax, xla and shardy source trees in ${JAX_TITAN_ROOT}/src at the
# tested revisions and applies the patches from third_party/patches, as in
# docs/getting_started.md section 2. Git only, so it runs on the login node;
# scripts/build_jax_rocm.sh (or scripts/slurm_job.sh) builds from these trees.
# A tree that exists is left as it is.
#
# jax and xla become detached worktrees of existing checkouts when JAX_REPO and
# XLA_REPO name them (default: sibling ../jax and ../xla, if they are git
# repositories), so no checkout of yours changes; otherwise they are cloned.
#
# Environment: JAX_TITAN_ROOT, JAX_REPO, XLA_REPO, JAX_REV, XLA_REV.
set -euo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
WORKSPACE=$(dirname "$REPO")
JAX_TITAN_ROOT=${JAX_TITAN_ROOT:-$WORKSPACE/.jax_titan}
PATCHES=$REPO/third_party/patches
SRC=$JAX_TITAN_ROOT/src
# Tested revisions (docs/getting_started.md). XLA must be at or after dfca2c9.
JAX_REV=${JAX_REV:-6b39d495e5}
XLA_REV=${XLA_REV:-d23c30982a}
JAX_REPO=${JAX_REPO:-$WORKSPACE/jax}
XLA_REPO=${XLA_REPO:-$WORKSPACE/xla}
mkdir -p "$SRC" "$JAX_TITAN_ROOT/logs"

# tree NAME LOCAL_REPO URL REV: sets CREATED=true when it made the tree.
# Not called from an if condition, where set -e would not stop on a failure.
tree() {
  local name=$1 local_repo=$2 url=$3 rev=$4 dst=$SRC/$1
  CREATED=false
  if [[ -e $dst ]]; then
    echo "$name: $dst exists ($(git -C "$dst" rev-parse --short HEAD)), leaving it"
    return 0
  fi
  if git -C "$local_repo" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$local_repo" cat-file -e "$rev^{commit}" 2>/dev/null || git -C "$local_repo" fetch -q origin
    git -C "$local_repo" worktree add -q --detach "$dst" "$rev"
    echo "$name: worktree of $local_repo at $rev"
  else
    git clone -q --filter=blob:none "$url" "$dst"
    git -C "$dst" checkout -q "$rev"
    echo "$name: cloned $url at $rev"
  fi
  CREATED=true
}

# patch_tree DIR PATCH...
patch_tree() {
  local dir=$1 p
  shift
  for p in "$@"; do
    git -C "$dir" apply --check "$PATCHES/$p" || { echo "$p does not apply to $dir" >&2; exit 1; }
    git -C "$dir" apply "$PATCHES/$p"
    echo "  applied $p"
  done
}

tree jax "$JAX_REPO" https://github.com/jax-ml/jax "$JAX_REV"
if $CREATED; then patch_tree "$SRC/jax" jaxlib-ifrt-ir-on-pjrt.patch; fi
tree xla "$XLA_REPO" https://github.com/openxla/xla "$XLA_REV"

# Shardy: the commit XLA pins, with the Shardy patches.
SHARDY_REV=$(sed -n 's/.*SHARDY_COMMIT = "\([0-9a-f]*\)".*/\1/p' "$SRC/xla/third_party/shardy/workspace.bzl")
[[ -n $SHARDY_REV ]] || { echo "no SHARDY_COMMIT in $SRC/xla/third_party/shardy/workspace.bzl" >&2; exit 1; }
tree shardy /nonexistent https://github.com/openxla/shardy "$SHARDY_REV"
if $CREATED; then
  patch_tree "$SRC/shardy" shardy-mpmd-oss-mlir-import.patch shardy-mpmd-parse-partitioning-options.patch
fi
echo "sources ready in $SRC"
