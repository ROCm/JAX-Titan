#!/usr/bin/env bash
# Submits scripts/slurm_job.sh as a one-node Slurm batch job with 8 GPUs,
# exclusive, logs in ${JAX_TITAN_ROOT}/logs/slurm/<JobID>.out and .err. Run on
# the login node, after scripts/setup_sources.sh. Prints the JobID on the last
# line.
#
#   slurm_submit.sh [OPTIONS] [-- TRAINER FLAGS...]
#
# Options (each overrides the matching JT_* setting below):
#   -p PARTITION   partition (JT_PARTITION; default: the cluster's default)
#   -q QOS         QOS (JT_QOS)
#   -A ACCOUNT     account (JT_ACCOUNT; default: your default account)
#   -t TIME        time limit (JT_TIME; default: 01:00:00; the first job builds for ~15 min)
#   -w NODE        a specific node
#   -J NAME        job name (default: $USER-jax_titan)
#   --setup-only   build the wheels and the venv, check the GPUs, and stop
#   -n, --dry-run  print the sbatch command without submitting
#
# Settings for your cluster go in a file named by JT_SITE_ENV (kept outside
# the repository), sourced here and by the job:
#   JT_PARTITION, JT_QOS, JT_ACCOUNT, JT_TIME   defaults for the options above
#   JT_GPU_REQUEST  how to ask for 8 GPUs (default: --gpus-per-node=8;
#                   some clusters need --gres=gpu:8 or --gres=gpu:<type>:8)
#   JT_SCRATCH, JT_IMAGE, TRAIN_TIMEOUT, BAZEL_CPUS, JAX_TITAN_ROOT   see slurm_job.sh
#   JT_SIGNAL       sends the batch shell SIGTERM before the time limit, so it
#                   cleans up (default: --signal=B:TERM@120; set it empty when
#                   sbatch does not support --signal)
#   SBATCH_EXTRA    more sbatch flags, one string
#
# Example: a 20-step Llama 3.1 8B pipeline on synthetic batches.
#   JT_SITE_ENV=~/my_cluster.env scripts/slurm_submit.sh -- --model=llama3.1-8b --data=synthetic --steps=20
set -euo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
if [[ -n ${JT_SITE_ENV:-} ]]; then
  [[ -r $JT_SITE_ENV ]] || { echo "JT_SITE_ENV=$JT_SITE_ENV is not readable" >&2; exit 2; }
  JT_SITE_ENV=$(readlink -f "$JT_SITE_ENV")
  # shellcheck source=/dev/null
  source "$JT_SITE_ENV"
fi
export JAX_TITAN_ROOT=${JAX_TITAN_ROOT:-$(dirname "$REPO")/.jax_titan}

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }

PART=${JT_PARTITION:-} QOS=${JT_QOS:-} ACCOUNT=${JT_ACCOUNT:-} TIME=${JT_TIME:-01:00:00}
NODE='' NAME=${USER}-jax_titan DRY=false JOB_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -p) PART=$2; shift 2 ;;
    -q) QOS=$2; shift 2 ;;
    -A) ACCOUNT=$2; shift 2 ;;
    -t) TIME=$2; shift 2 ;;
    -w) NODE=$2; shift 2 ;;
    -J) NAME=$2; shift 2 ;;
    --setup-only) JOB_ARGS+=(--setup-only); shift ;;
    -n|--dry-run) DRY=true; shift ;;
    -h|--help) usage ;;
    --) shift; break ;;
    *) echo "unknown option $1 (trainer flags go after --)" >&2; usage 1 >&2 ;;
  esac
done
JOB_ARGS+=("$@")

for d in jax xla shardy; do
  [[ -d $JAX_TITAN_ROOT/src/$d ]] || { echo "missing $JAX_TITAN_ROOT/src/$d: run scripts/setup_sources.sh first" >&2; exit 2; }
done

# shellcheck disable=SC2206  # JT_GPU_REQUEST, JT_SIGNAL and SBATCH_EXTRA are split into flags on purpose.
ARGS=(-N1 --exclusive --parsable ${JT_GPU_REQUEST:---gpus-per-node=8}
  -J "$NAME" -t "$TIME" ${JT_SIGNAL-'--signal=B:TERM@120'})
[[ -n $PART ]] && ARGS+=(-p "$PART")
[[ -n $QOS ]] && ARGS+=(--qos="$QOS")
[[ -n $ACCOUNT ]] && ARGS+=(-A "$ACCOUNT")
[[ -n $NODE ]] && ARGS+=(-w "$NODE")
# shellcheck disable=SC2206
ARGS+=(${SBATCH_EXTRA:-})

LOGS=$JAX_TITAN_ROOT/logs/slurm
mkdir -p "$LOGS"
# Pass the job's settings in the command itself rather than relying on the
# scheduler to export the submitting shell's environment.
ENVS=(JAX_TITAN_ROOT="$JAX_TITAN_ROOT")
for v in JT_SITE_ENV JT_IMAGE JT_SCRATCH TRAIN_TIMEOUT BAZEL_CPUS; do
  [[ -n ${!v:-} ]] && ENVS+=("$v=${!v}")
done
CMD=(sbatch "${ARGS[@]}" -o "$LOGS/%j.out" -e "$LOGS/%j.err"
  --wrap "env $(printf '%q ' "${ENVS[@]}") bash $(printf '%q ' "$REPO/scripts/slurm_job.sh" "${JOB_ARGS[@]}")")

if $DRY; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

# Submit from /tmp: the job's working directory must exist on the node.
out=$(cd /tmp && "${CMD[@]}")
JOBID=${out%%;*}  # --parsable prints JOBID or JOBID;CLUSTER
[[ $JOBID =~ ^[0-9]+$ ]] || { echo "sbatch failed: $out" >&2; exit 1; }
echo "submitted job $JOBID"
echo "  log:    $LOGS/$JOBID.out (and .err)"
echo "  status: squeue -j $JOBID; scontrol show job $JOBID | grep -E 'JobState|Reason'"
echo "  cancel: scancel $JOBID"
echo "$JOBID"
