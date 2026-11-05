# SPUR cluster guide

SPUR is AMD's internal cluster of nodes with 8 AMD Instinct MI355X GPUs each,
run by the `spur` scheduler. This guide covers reaching the scheduler, getting
a node, running commands on it, keeping it and releasing it. It merges and
translates two guides from the internal RL runbook collection: the guide for
the v2 pool and the older node access guide for the original pool (see
[source-runbooks.md](source-runbooks.md)).
[jax-titan-on-spur.md](jax-titan-on-spur.md) builds on it.

Each section says how it was tested:

- **Tested on v2 (2026-09-16):** the source ran it on the v2 pool on that date.
- **Tested on the old pool:** the source ran it on the original pool. It
  records no date, except for the multi-node runs, which were on 2026-08-05.
  The pools differ in many places, so do not assume these steps work on v2.
- **Untested:** nobody ran it. Treat it as a lead, not a result.

The two pools often need opposite habits. On the old pool, a GPU job requested
without a real terminal stays on hold forever, so a person has to run
`spur alloc` in a terminal. On v2, `sbatch` gets a GPU node directly.

## Cluster facts

Every cluster-specific value in these runbooks is in this table; the other
sections and files refer to them through the variables in the last column.

| Fact | Value | Variable |
|---|---|---|
| v2 controller | `http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817`. Other v2 host names, for example `crs-m2m-cpu-spur-v2-005`, refuse connections. | `SPUR` |
| Old-pool controllers, also the login nodes' default | `http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817`, and the same on `-spur-006` and `-spur-007` | `SPUR_OLD` |
| Login nodes, used for both pools | `crs-m2m-cpu-spur-0XX.crusoe.amd.com` | |
| v2 compute nodes | `crsuse2-m2m-v2-0XX`, about 43 nodes | `NODE`, `NODE_RE='crsuse2-m2m-v2-[0-9]+'` |
| Old-pool compute nodes | `crsuse2-m2m-XXX` | `NODE_RE='crsuse2-m2m-[0-9]+'` |
| v2 partition | `default`, `TIMELIMIT=infinite` | `PART` |
| v2 nodes that are not in `default` | `crsuse2-m2m-v2-045` to `-049` (partition `k0s`); `crsuse2-m2m-v2-050` (reserved for Kubernetes, still shows as idle) | |
| Example team pool on v2 | `crsuse2-m2m-v2-035` and `-037` to `-043` (no `-036`), dedicated to the source's team | |
| Old-pool partition, account and QOS | Partition `amd-spur`. Account `amd-aifw-dev` with default QOS `amd-aifw-dev-qos` (`GrpTRES node=19`, shared by everyone in the account). `amd-burst-qos`: `node=128`, priority 100 against 10000, `Preempt=off`. | `<partition>`, `<account>`, `<burst QOS>` |
| SSH | Compute nodes allow only `AllowUsers ubuntu root`. v2 login nodes refuse ssh as well. | |
| Node hardware | `gpu:mi355x:8` (gfx950, 288 GB per GPU), `CPUTot=236`, about 2.8 TB RAM (`free -g` shows 2751G) | |
| Idle GPU memory | About 298 MB per GPU in `rocm-smi` | |
| Shared home | `/home/$USER`: NFS, the same files on the login nodes and the compute nodes of both pools. A 10 TB volume shared by the whole group, with about 2.5 TB free in August 2026. | `$HOME` on the login node |
| Node-local scratch | `/mnt/m2m_nobackup`: 28 TB NVMe per node; also holds the docker root directory | `SCRATCH` |
| `spur exec` environment | v2: `HOME=/opt/spur` (formerly `/root/spur`, not writable), `PWD=/`. Old pool: root by default, working directory `/`. | |
| Login profile scripts | `/etc/profile.d/spur.sh` and `spur.csh` set `SPUR_CONTROLLER_ADDR` to the old pool | |
| Routable inter-node interface | `ens3`, 200 Gb/s, TCP | `NET_IF` |
| Interfaces that do not route between nodes | `docker0`, `br-*` (172.x) | |
| RDMA NICs | 8x `ionic_*`, for example `ionic_0` (AMD Pensando AINIC, 400 Gb/s RoCE, IPv6 `fc01:` addresses only, each NIC on a different /64 per node) and 1x `mlx5_0` (200 Gb/s frontend) | `IB_HCA` |
| Host ionic verbs provider | `/etc/libibverbs.d/ionic.driver` and `/usr/lib/x86_64-linux-gnu/libionic.so.*` (rdma-core v54) | `IONIC_DRIVER`, `IONIC_LIB` |
| Host RCCL network plugin | `/etc/profile.d/99-rccl-anp.sh` sets `NCCL_NET_PLUGIN=anp`; the plugin is `/opt/rocm-7.0.1/lib/librccl-net.so`. AINIC packages exist only for Ubuntu 24.04 (`/opt/amd/ainic/deb-repo`). | |
| Scheduler | `spur` with Slurm-style aliases (`sbatch`, `srun`, `squeue`, `scancel`, `sinfo`, `sacct`, `sattach`). Version 0.7.0 when the multi-node notes were written. | |

Set the variables you need in your shell on the login node:

```bash
SPUR=<v2 controller URL>
PART=<v2 partition>
NODE=<v2 compute node>
NODE_RE='<compute node name pattern>'
SCRATCH=<node-local scratch path>
```

## 1. Rules for agents

These rules apply to people too. Read them before running anything.

1. **Use only the JobID you were given.** Run `spur exec`, `spur show job` and
   log reads against that job only. If it is not `RUNNING`, is not yours or
   does not exist, stop and report. Do not switch to another job you see in
   `squeue`.
2. **Do not allocate on your own.** Never run `sbatch`, `spur alloc` or
   `spur run` to get nodes unless the person you work for asked you to.
3. **Never ssh to compute nodes.** Use `spur exec` (section 5). On v2 the
   login nodes refuse ssh as well; work from your usual login node.
4. **Never `scancel` a job that is not yours.** A JobID someone gave you is
   lent to you: cancel it only when its owner asks.
5. **Release what you allocate.** Prefer task mode, which releases the node
   when the script exits. In hold mode, remove your containers and cancel the
   job as soon as you are done.
6. **Leave other people's resources alone.** Do not change other users' home
   directories, and do not kill processes that do not belong to your job.

Fix the JobID once and refer to it from then on:

```bash
JOBID=<JobID>
spur show job --controller "$SPUR" "$JOBID" | head -20    # state, nodes, time limit
```

## 2. Connect to the scheduler

**Tested on v2 (2026-09-16).**

v2 is a separate scheduler instance. `sinfo` and `squeue` show whatever
cluster the controller address points at, and switching clusters only means
switching that address. You do not need v2's own login nodes (you cannot log
in to them anyway): work from the usual login node.

The login nodes' default controller is the old pool, so point every command
at v2 explicitly. Every `spur` subcommand and every Slurm-style alias accepts
`--controller`, which is the most reliable way:

```bash
sinfo     --controller "$SPUR"
squeue    --controller "$SPUR" -u "$USER"
sbatch    --controller "$SPUR" ...
scancel   --controller "$SPUR" <JobID>
spur exec --controller "$SPUR" <JobID> bash -c '...'
spur show job --controller "$SPUR" <JobID>
```

An environment variable also works, but login shells can overwrite it without
an error:

```bash
export SPUR_CONTROLLER_ADDR="$SPUR"
```

Some login nodes have had a profile script (Cluster facts) that sets the
variable to the old pool unconditionally. Wrapping a command in `bash -l` or
`bash -lc` then sends it back to the old pool: `sinfo | grep -cE "$NODE_RE"`
with the v2 pattern counts zero nodes and v2 looks empty, while you are asking
the wrong cluster. If you must use the variable, wrap commands in `bash -c`,
not `bash -l`.

The `-l` in `spur exec <JobID> bash -lc '...'` is a different matter and is
fine: that `bash` runs on the compute node, after the client has read the
controller address.

The v2 controller address in Cluster facts is the only one that works;
another v2 host name tried as the controller refused connections.

### Old pool

**Tested on the old pool.** The login nodes' profile scripts set
`SPUR_CONTROLLER_ADDR` to the old pool, so three checks are enough:

```bash
hostname -f                      # which login node you are on
echo "$SPUR_CONTROLLER_ADDR"     # must not be empty
squeue -u "$USER"                # any output means the scheduler answers
```

If you run commands non-interactively from outside, as `ssh <login node>
"<command>"`, the login shell can be csh or tcsh:

- Set variables with `setenv A B`, not `export A=B`.
- Do not use `2>&1`, `2>/dev/null` or pipes in the command string; csh fails
  with `Ambiguous output redirect`.
- The profile is not loaded, so set the controller yourself:
  `setenv SPUR_CONTROLLER_ADDR <old-pool controller URL>`.

`echo $SHELL` can print tcsh even when the agent's commands run in bash; go by
the actual error messages.

## 3. Look before you allocate

**Tested on v2 (2026-09-16).**

```bash
# Overview. The v2 partition has no time cap.
sinfo --controller "$SPUR"

# Your jobs. %L is the time left, %l the limit.
squeue --controller "$SPUR" -u "$USER" -o '%.8i %.22j %.2t %.12M %.12L %.14l %R'

# Node states. Only idle nodes can be allocated; alloc means the whole node is
# taken, mix means part of it.
sinfo --controller "$SPUR" -N -o '%N|%t|%G' | grep -E "$NODE_RE"

# Who holds the nodes and under which job names, to judge the competition.
squeue --controller "$SPUR" -o '%.8i %.12u %.24j %.6D %.20R %.12T' | grep -E "$NODE_RE|USER"

# One node's resources.
spur show node --controller "$SPUR" "$NODE"

# Final state and exit code of past jobs; task mode reports success here.
sacct --controller "$SPUR"
```

Three things behave against intuition:

- **Idle nodes are taken within about 100 seconds.** Several teams run jobs
  on v2; one job got a node about 100 seconds after it was submitted. Take an
  idle node early rather than assuming it will wait for you.
- **`idle` in `sinfo` does not mean you can get the node.** On 2026-09-16, a
  node was `State=IDLE` with an empty `Reason` in both `sinfo` and
  `spur show node`, but a submission to it stayed pending with
  `(ReqNodeNotAvail, Reserved for Kubernetes cluster)`. Nodes can carry
  reservations that `spur show node` does not show, and some nodes belong to
  another partition (Cluster facts). Always check that your job reaches `R`.
- **A high-priority pending job is not necessarily a competitor.** A job with
  priority 11000 stayed pending for 20 hours while 8 nodes were idle. Such
  jobs usually wait for one specific node (job names like `hold-<node>`) or
  are dead. Look at the job name and how long it has been pending.

## 4. Allocate

Skip this section if you were given a JobID.

| | Task mode (recommended) | Hold mode |
|---|---|---|
| What you submit | Your job script | `sleep <long time>` |
| When it ends | When the script exits; the node goes back to the pool | Only when you `scancel` it |
| Good for | Training, evaluation, batch runs, CI | Debugging, interactive work, repeated container access |
| Risk of never releasing | None | High: the GPUs stay idle and taken |

On v2, do not pass `-A` in either mode (section 4.3).

### 4.1 Task mode

**Tested on v2 (2026-09-16).** Hand the command to `sbatch`. When the script
exits, the job ends and the node returns to the pool at once; you do not need
to, and should not, `scancel` it.

```bash
SHARED=$HOME/jobs            # must be on the shared home; see the pitfalls below
mkdir -p "$SHARED/logs"
sbatch --controller "$SPUR" --parsable \
  -J "$USER-train" -p "$PART" \
  -N1 -w "$NODE" \
  --gpus-per-node=8 --exclusive -c 64 \
  -t 1-00:00:00 \
  -o "$SHARED/logs/%j.out" -e "$SHARED/logs/%j.err" \
  "$SHARED/task.sh" arg1 arg2
```

Instead of a script file you can use `--wrap` (not both):

```bash
sbatch --controller "$SPUR" --parsable -J "$USER-smoke" -p "$PART" \
  -N1 -c8 -t 00:30:00 -o "$SHARED/logs/%j.out" \
  --wrap 'set -euo pipefail; cd /shared/path/to/repo; python train.py'
```

What the source observed:

- A job reached `R` about 1 second after submission. A `--wrap 'sleep 5'` job
  was still in the queue at second 6 and gone at second 7, without any cleanup.
- `-t` is a safety limit, not a reservation: a task that ends early releases
  the node early. The job is killed when `-t` runs out, so leave margin.
- After the job, `sacct` tells whether it succeeded. Only `State=COMPLETED`
  with `ExitCode=0:0` is a success:

  ```bash
  sacct --controller "$SPUR" | grep <JobID>
  #  <JobID>  <name>  <user>  <partition>  COMPLETED  00:00:04  1  0:0
  ```

Four pitfalls, all observed:

- **Write logs to a shared directory.** `-o /tmp/x.out` produced no file on
  the login node: it was written to the compute node's own `/tmp`. Likewise,
  the job's `$PWD` (by default the path of the directory you submitted from)
  is resolved on the compute node and can be a different disk with the same
  path. Always give `-o` and `-e` shared paths, with `%j` so that runs do not
  overwrite each other.
- **Everything the script uses must be on shared storage.** The script itself
  need not be: `sbatch` ships its content with the job (it lands as
  `/var/spool/spur/job<id>/spur_job.sh` on the node), so a script in the
  login node's `/tmp` runs and receives its arguments in `$1`. But files it
  sources, reads or writes must be shared, or the job fails halfway.
- **Without `-c` the script gets one CPU.** The job script runs in a cgroup:
  with `-c1`, `nproc` in the script printed 1; with `-c8`, 8. Data loaders
  stall on one CPU, so always pass `-c`. This differs from `spur exec` into a
  hold-mode job, which printed `nproc=236` because it does not run in the
  job's cgroup, so do not infer a task's CPUs from `spur exec`.
  **Untested:** whether `--exclusive` without `-c` gives 1 or 236 CPUs. Pass
  `-c` explicitly, and print `nproc` at the start of the task's log.
- **Without `--exclusive`, other people's CPU jobs land on the node.** With
  only `--gpus-per-node=8` you get the 8 GPUs, but the rest of the CPUs stay
  free for others, and the node shows `mix` instead of `alloc`.

### 4.2 Hold mode

**Tested on v2 (2026-09-16).** Use it to enter containers repeatedly while
debugging, or to keep a service running on the node. The price is that you
must release it yourself.

```bash
cd /tmp      # the source submits from /tmp; the job's working directory is resolved on the node
sbatch --controller "$SPUR" --parsable \
  -J "$USER-hold" -p "$PART" \
  -N1 -w "$NODE" --gpus-per-node=8 --exclusive \
  -t 7-00:00:00 --wrap "sleep 604800"
```

Then confirm:

```bash
squeue --controller "$SPUR" -u "$USER"                    # wait for ST=R; note NODELIST
spur show job --controller "$SPUR" <JobID> | head -20     # expect TresPerNode=gpu:8/node
```

Section 5 explains how to work on the node, section 7 how to renew and
section 8 how to release.

### 4.3 Notes for both modes

**Tested on v2 (2026-09-16).**

- **Do not pass `-A`.** v2's user table is a strict subset of the old pool's:
  many people with accounts on the old pool have none on v2, and `-A` is then
  rejected:

  ```
  Error: job submission failed
  Caused by:
      code: 'Client specified an invalid argument', message: "user '<you>' has no account associations.
      Contact your cluster admin to run: sacctmgr add user name=<you> account=<account>"
  ```

  The scheduler requires an association only when an account is given;
  without `-A` the job takes the account-less path and needs no `-q` either
  (its `QOS` field is empty).
- **Account-less jobs have lower priority:** `Priority=1000`, against 11000
  for jobs with an account, so jobs with an account go first when both want
  the same idle node. An administrator can add the association with the
  command from the error message, but this does not block you: jobs without
  `-A` run today.
- **There is no `--test-only`.** `spur`'s `sbatch` does not accept it, so
  check the node's state before submitting (section 3) and confirm `R`
  afterwards.
- **Do not judge exclusive use from `Exclusive=` in `spur show job`.** A job
  that had the whole node showed `Exclusive=0`. Check that `CPUAlloc` equals
  `CPUTot` in `spur show node`, or that `sinfo` shows the node as `alloc`
  rather than `mix`.
- **`CPUs/Task=1` in the job details does not tell how many CPUs a container
  can use;** see the `-c` pitfall in section 4.1.
- **Only `sinfo -n` and `spur show node` tell whether a node exists.** A node
  whose name resolved in DNS and whose port 22 was open was unknown to the
  scheduler (`sinfo -n` returned `n/a`) and was registered days later.

  ```bash
  sinfo --controller "$SPUR" -n "$NODE"
  ```

### 4.4 Team nodes

**Tested on v2 (2026-09-16).** If your team has dedicated v2 nodes, request
them by name with `-w <node>` instead of competing in the shared pool. Cluster
facts lists the source team's pool as an example; use your own list.
Dedicated does not mean free: colleagues often hold the whole pool with jobs
of 3 to 5 days, and on 2026-09-16 all eight of the source team's nodes were
`alloc`. Look before you submit (section 3).

### 4.5 Old pool

**Tested on the old pool.** Look up your own account, QOS and partitions; do
not copy someone else's:

```bash
spur accounts show user where name="$USER"    # your accounts, QOS and defaults
spur nodes                                    # partitions (the default has a *); same as sinfo
spur queue                                    # global queue; same as squeue
scontrol show job "$JOBID"                    # job details
```

Fill the account and partition into `-A <account> -p <partition>`. The QOS is
usually `<account>-qos`, but `spur alloc` does not accept `-q`, so it uses the
default QOS.

GPU jobs are picky about how they are requested:

- `spur alloc` in a **real terminal** reliably gets an 8-GPU node:

  ```bash
  spur alloc -N1 -t 1-00:00:00 -A <account> -p <partition> -G 8
  ```

  It needs a tty, accepts no trailing command and opens an interactive shell;
  `exit` releases the node.
- `sbatch --wrap "..."` with GPUs (`-G 8` or `--gres gpu:8`) stays
  `PENDING (JobHoldMaxRequeue)` forever, and so does `spur alloc` run through
  `ssh -tt` with piped stdin. This is not a per-account limit: requesting GPUs
  without a real tty does not work on this pool. So a person allocates GPU
  nodes in a real terminal and hands the JobID to the agent.
- CPU-only placeholders through `sbatch` work:

  ```bash
  sbatch --parsable -J myjob -A <account> -p <partition> -N1 -t 04:00:00 --wrap "sleep 14400"
  ```

- You can hold several 8-GPU jobs at once; there is no limit of one per
  account.

For two or more nodes, the first limit you hit is the QOS group quota, not
free nodes. The quota counts nodes for the whole account (Cluster facts), so
teammates use it up:

```bash
spur accounts show qos            # the GrpTRES column
squeue -t RUNNING -o '%q %D' | tail -n +2 | awk '$1=="<your default QOS>"{s+=$2} END{print s}'
```

When it is used up, `-N 4` stays `PENDING (QOSGrpNodeLimit)` while
`spur nodes` shows dozens of idle nodes. `spur alloc` cannot change the QOS;
`spur run` can:

```bash
spur run -q <burst QOS> -N 4 --gpus-per-node=8 \
  -t 1-00:00:00 -A <account> -p <partition> --pty bash -l
```

Each of these details was learned the hard way:

- Use `--gpus-per-node=8`, not `-G 32`. `-G` counts GPUs for the whole job,
  and the scheduler may spread them over the nodes in any way.
- Keep `--pty`. Without a real tty the GPU request falls into
  `JobHoldMaxRequeue`; the queue permanently holds 30 or more such jobs, which
  hold no resources and never run.
- Do not run it in a shell that is already inside an allocation. With
  `SLURM_JOB_ID` set, `spur run` becomes a job step of that job: `-N 4` and
  `-q` are ignored, it prints `dispatched to node <one node>`, and no new
  JobID appears. Check first:

  ```bash
  env | grep -E 'SLURM_JOB_ID|SLURM_NNODES'    # must print nothing
  ```

- Success prints `Pending job allocation <JobID>...` and then
  `running on <node list>`. Confirm the shape:

  ```bash
  scontrol show job "$JOBID" | head -12    # NumNodes=4, TresPerNode=gpu:8/node, ReqGPUs=32
  ```

The burst QOS has low priority and queues behind the default QOS, but it has
`Preempt=off`, so its jobs are not preempted.

## 5. Run commands on the node

**Tested on v2 (2026-09-16) and on the old pool.** The only way in is
`spur exec`; the controller forwards the command to the node and runs it in
the job's environment there.

```bash
spur exec --controller "$SPUR" "$JOBID" bash -c 'hostname; whoami; pwd'
```

- **ssh does not work** (Cluster facts). On v2 both the compute nodes and the
  login nodes answer `Permission denied (publickey,password)`, so you always
  work from your usual login node with the v2 controller.
- **`spur exec` checks the job owner.** You cannot enter someone else's job:
  `user <you> cannot exec into job owned by <other>`. If a colleague holds
  the node, wait.
- **`spur exec <JobID> bash` without `-c` hangs.** `spur exec` allocates no
  tty, so bash starts and waits for stdin. Run commands with `bash -c "..."`;
  for an interactive shell, a person opens one with `spur alloc` (tested on
  the old pool; not among the tested v2 commands).
- **`HOME` is not your home directory** (Cluster facts), and `PWD` is `/`.
  Use absolute paths in every command. (On v2 it used to be a directory that
  was not writable, and every command printed
  `bash: <that directory>/.bash_profile: Permission denied`; that no longer
  happens and is harmless if it does.)
- **Give docker a writable configuration directory before using it:**

  ```bash
  mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
  ```

Health check (the GPUs are AMD ROCm GPUs; there is no `nvidia-smi`):

```bash
spur exec --controller "$SPUR" "$JOBID" bash -c \
  'hostname; nproc; uptime; free -h; rocm-smi'
```

`rocm-smi --showpids` lists the processes on the GPUs (old pool). `ps` works
on v2. If it prints `Error, do this: mount -t proc proc /proc`, as
it did on the old pool, mount `/proc` first:

```bash
spur exec --controller "$SPUR" "$JOBID" bash -c \
  'mount -t proc proc /proc; ps -eo pid,user,pcpu,pmem,etime,args --sort=-pcpu | head -n 20'
```

### Long-running commands

Keep code and logs on the shared home, where the login node can read them.
The old node access guide gives this template:

```bash
PROJ=$HOME/<your project>
LOGDIR=$HOME/logs; mkdir -p "$LOGDIR"
spur exec "$JOBID" bash -lc "cd $PROJ && ./run.sh"
spur exec "$JOBID" bash -lc "cd $PROJ && nohup python train.py > $LOGDIR/train_$JOBID.log 2>&1 &"
spur exec "$JOBID" bash -lc "tail -n 50 $LOGDIR/train_$JOBID.log"
```

The source does not say whether the `nohup ... &` line survives after
`spur exec` returns, and in `srun` steps (section 6) such background processes
were killed when the step ended. On the old pool, the RL runbooks ran their
long jobs in a container with `docker exec -d`, which survived both the
`spur exec` call and the `srun` steps that started it. The same on v2
(untested there):

```bash
spur exec --controller "$SPUR" "$JOBID" bash -c \
  "mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
   docker exec -d <container> bash -lc '<command> > <shared log> 2>&1'"
```

## 6. Multi-node jobs: reaching every node

**Tested on the old pool (4 nodes x 8 GPUs, 2026-08-05). Untested on v2.**
On v2 nobody has checked that `spur exec` reaches only the head node or that
the `srun` route below works.

In a multi-node job, only one of three routes works for every node:

| Route | Result |
|---|---|
| `ssh <node>` | `Permission denied (publickey)` (Cluster facts) |
| `spur exec <JobID>` | Reaches **only the head node**. It has no node argument; the controller always forwards to the head. |
| `srun --jobid <JobID> --overlap -w <node> <command>` | Works, but **silently drops the output when stdin is not a tty** |

The third route is a trap. spur 0.7.0 prints one line,
`spur: warning: raw mode unavailable (stdin is not a TTY)`, and then the
command seems to do nothing: `hostname` sometimes prints, `ls /` and
`bash -c 'echo hi'` print nothing, and the node looks broken. Wrapping the
call in `script -qec` to provide a pty fixes it. The wrapper `~/nx.sh` passes
the snippet through the shared home, which every node sees:

```bash
cat > ~/nx.sh <<'EOF'
#!/usr/bin/env bash
# Run a bash snippet on one specific node of a spur multi-node job.
#
# spur exec <jobid> only ever reaches the job's head node, and srun job steps
# silently produce nothing unless stdin is a real tty (spur 0.7.0 prints
# "raw mode unavailable" and drops the output), hence the script(1) wrapper.
# The snippet travels through $HOME because that is shared NFS across nodes.
set -uo pipefail
: "${JOBID:?set JOBID to the spur job id}"
NODE="${1:?usage: nx.sh <node> [command...]}"
shift
D="$HOME/.nx"; mkdir -p "$D"
F="$D/cmd_${NODE}_$$_$RANDOM.sh"
if [ $# -gt 0 ]; then printf '%s\n' "$*" > "$F"; else cat > "$F"; fi
script -qec "srun --jobid $JOBID --overlap -w $NODE -N1 -n1 bash -l $F" /dev/null
rc=$?
rm -f "$F"
exit $rc
EOF
chmod +x ~/nx.sh
```

On v2 (untested), `srun` would also need the v2 controller: change the call to
`srun --controller $SPUR --jobid $JOBID ...`.

Check every node; each should print its host name and 8 GPUs:

```bash
export JOBID=<JobID>
for n in $(scontrol show job "$JOBID" | grep -oE "$NODE_RE" | sort -u); do
  printf "%s: " "$n"
  ~/nx.sh "$n" 'hostname; rocm-smi --showid 2>/dev/null | grep -oE "GPU\[[0-9]+\]" | sort -u | wc -l'
done
```

The output contains a `^@` left by `script`; filter it with `sed 's/\^@//'`.
`nx.sh` runs on the docker host side of the node. To run something in a
container, call docker through it:
`~/nx.sh <node> 'docker exec <container> bash -lc "..."'`.

Ray and torchrun handed out ranks contiguously by node. In a 4x8 job:

| Node | Ranks |
|---|---|
| Head (the one `spur exec` reaches) | 0-7 |
| Second | 8-15 |
| Third | 16-23 |
| Fourth | 24-31 |

This helps find the node of a rank. It is observed behavior, not a guarantee:
code that depends on it, for example to keep a parallel group inside one
node, should check it.

Node-local scratch is separate on each node, so anything every node reads
must be on the shared home. To see whether a path is shared, compare it
across nodes:

```bash
for n in <node1> <node2>; do printf "%s: " "$n"; ~/nx.sh "$n" "ls $SCRATCH | head -3"; done
```

[multi-node-notes.md](multi-node-notes.md) has the rest of what the
multi-node RL runs learned.

## 7. Renew

**Tested on v2 (2026-09-16).** Raise the time limit of the running job in
place; the node never leaves your hands, and no new job or `scancel` is
needed:

```bash
spur scontrol update --controller "$SPUR" JobId=<JobID> TimeLimit=30-00:00:00
squeue --controller "$SPUR" -u "$USER" -o '%.8i %.12L %.14l'    # confirm it took effect
```

The source ran it with 2 hours 14 minutes left: `TimeLimit` went from
`7-00:00:00` to `30-00:00:00`, `EndTime` moved 23 days later and the job kept
running. This is the preferred way to renew.

- **`scontrol` on its own is `command not found` on the login node.** Only
  `spur` is installed and the Slurm commands are its aliases, so use
  `spur scontrol ...` (`spur --help` lists the aliases at the end).
- **Raising a limit usually needs administrator rights; do not count on it.**
  It worked for an account-less job, for reasons nobody looked into. Check
  `TIME_LIMIT` and `TIME_LEFT` with `squeue` every time.
- **Only raise it.** Lowering the limit shortens the job at once, with no
  undo.

To move to another node, **submit the new job first, then cancel the old
one**; the reverse order exposes the node, and an idle v2 node can be taken
within about 100 seconds. Done in this order, the takeover finished within 4
seconds and the node was never free. Moving is not a way to renew: dedicated
nodes are often all `alloc`, a new job competes with `Priority=11000` jobs
(account-less jobs have 1000), and node-local scratch does not move with you.

## 8. Release

**Tested on v2 (2026-09-16).** Task mode needs nothing: the node is released
when the script exits. In hold mode:

1. **Remove your containers first,** or the next user will find the GPUs
   taken. Then check with `rocm-smi` on the node that GPU memory is back at
   the idle level of about 298 MB per GPU:

   ```bash
   spur exec --controller "$SPUR" <JobID> bash -c \
     'export DOCKER_CONFIG=/tmp/.docker; docker rm -f <container>; sleep 10; rocm-smi --showmeminfo vram'
   ```

   This happened: after a task had finished, every GPU still had about
   90.9 GB in use, which cut the next user's memory budget.
2. Cancel the job:

   ```bash
   scancel --controller "$SPUR" <JobID>     # or: spur cancel --controller "$SPUR" <JobID>
   ```

Only cancel your own jobs, or jobs whose owner asked you to. A JobID someone
gave you is lent to you. Cancel the placeholder and interactive jobs you
allocated as soon as you are done, rather than holding GPUs.

## 9. Storage

**Tested on v2 (2026-09-16), with the multi-node point from the old pool.**

- **The shared home is the same on v2 and the old pool;** the source compared
  files one by one. Environments built for the old pool need no rebuild, and
  checkouts on the login node are visible on the nodes. Keep job scripts,
  logs, checkpoints and outputs there. It is shared by the whole group and has
  limited free space (Cluster facts): do not put hundreds of GB there without
  checking.
- **Node-local scratch is per node** (it also holds the docker root
  directory), so a new node means rebuilding outputs and pulling images again.
- **`/tmp` is per node**, and a compute node's `/tmp` is a different disk from
  the login node's `/tmp` with the same path. Do not pass anything you will
  read later through `/tmp`.
- **Node-local storage does not reliably survive between jobs either.** Node-
  local scratch was wiped once; containers and tens of GB of weights were
  gone, while code and logs on the shared home were intact.
- **In multi-node jobs, every node sees a different node-local scratch**
  (tested on the old pool), so models, data and anything all nodes read go on
  the shared home.

## 10. Reliability

**Tested on v2 (2026-09-16), except where marked.**

- **Do not trust `EndTime`.** A job that claimed 5 more days disappeared 5
  days early, its log cut off without an error. Check that the job exists
  with `squeue --controller "$SPUR" -u "$USER"` before every action.
- **Preemption is untested.** Both v2 QOS have `PreemptMode=off`, but
  account-less jobs do not use either QOS, so this does not mean v2 never
  preempts.
- **Save checkpoints of long jobs yourself, on the shared home** (section 9).

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `sinfo` finds no v2 nodes; v2 looks empty | The controller points at the old pool, the default; `bash -l` can reset the variable | Use `--controller "$SPUR"` (section 2) |
| `failed to connect to spurctld ... Connection refused` | No controller set, or another v2 host name | Use the v2 controller from Cluster facts; on the old pool set `SPUR_CONTROLLER_ADDR` (`setenv` in csh) |
| `job submission failed ... has no account associations` | `-A` given; v2 has no association for you | Submit without `-A` (priority 1000) |
| Pending with `(ReqNodeNotAvail, Reserved for Kubernetes cluster)`, `sinfo` says idle | A reservation `spur show node` does not show | Pick another node (section 3) |
| 8 GPUs allocated, but the node is `mix` and others' CPU jobs land on it | `--exclusive` missing | Add `--exclusive` |
| `Exclusive=0` in `spur show job` | The field is unreliable | Check `CPUAlloc` = `CPUTot` in `spur show node`, or `alloc` in `sinfo` |
| Task runs very slowly; data loading stalls | No `-c`: the script gets 1 CPU | Pass `-c`; do not infer CPUs from `spur exec` |
| `-o` log missing on the login node | Written to the compute node's `/tmp` | Shared paths with `%j` |
| Task fails halfway with a missing file | It reads node-local paths; only the script is shipped | Keep data and dependencies on the shared home |
| A finished task is gone from `squeue`; did it succeed? | Tasks leave the queue when they end | `sacct --controller "$SPUR"`: `COMPLETED` and `0:0` |
| Job close to its time limit | It is killed at `-t` | `spur scontrol update ... TimeLimit=...` (section 7) |
| `scontrol: command not found` | Only `spur` is installed | `spur scontrol ...` |
| `sbatch --test-only` fails with `unexpected argument` | Not supported | Check node state, submit, confirm `R` |
| A node resolves in DNS and has port 22 open but `sinfo` does not know it | Not registered with the scheduler yet | Trust only `sinfo -n` and `spur show node` |
| `ssh <node>` gives `Permission denied` | ssh is blocked (Cluster facts) | `spur exec` (section 5) |
| `cannot exec into job owned by <user>` | `spur exec` checks the job owner | Use your own job; wait for the node |
| `spur exec <JobID> bash` hangs without a prompt | No tty | `bash -c "..."`; a person uses `spur alloc` for a shell |
| `$HOME` is wrong; relative paths are not found | `spur exec` has its own `HOME` and `PWD=/` | Absolute paths |
| `ps` prints `Error, do this: mount -t proc proc /proc` | `/proc` not mounted | `mount -t proc proc /proc` |
| `nvidia-smi: command not found` | AMD ROCm GPUs | `rocm-smi` |
| Old pool: `Unknown option '-lc'` or `Ambiguous output redirect` | The remote shell is csh | `setenv`; no redirects in the command string |
| Old pool: GPU job pending forever with `JobHoldMaxRequeue` | GPUs requested without a real tty (`sbatch --wrap`, faked tty); not an account limit | A person runs `spur alloc ... -G 8` in a terminal and hands over the JobID |
| Old pool: `spur alloc` reports `unexpected argument '-q'` or `'sleep'` | It takes neither `-q` nor a trailing command | Drop `-q`; use `sbatch` for scripted placeholders |
| Old pool: `Invalid account or account/partition combination` | `-A`/`-p` copied from someone else | `spur accounts show user where name="$USER"` |
| Old pool: `-N 2` or more pending with `QOSGrpNodeLimit`, many idle nodes | The account-wide QOS quota is used up | `spur accounts show qos`; another QOS with `spur run -q` (section 4.5) |
| Old pool: `spur run` prints `dispatched to node <one node>`, no new JobID | Run inside an allocation; became a job step | A clean shell where `env \| grep SLURM_JOB_ID` prints nothing |
| Old pool: `srun --overlap -w <node>` prints only `raw mode unavailable` | stdin is not a tty; spur 0.7.0 drops the output | `nx.sh` (section 6) |
| Old pool: `spur exec` always reaches the same node of a multi-node job | It only reaches the head node | `nx.sh` for the other nodes |
| Node-local scratch differs or is empty across nodes | It is node-local | Shared data on the shared home (section 9) |
| A job disappeared; its log stops without an error | Jobs can end early; preemption is untested | Check `squeue` before acting; checkpoint to the shared home |
| Outputs gone after moving to another node | Node-local storage | Keep what you need on the shared home |

## 12. Quick reference

```bash
SPUR=<v2 controller URL>
PART=<v2 partition>
SHARED=$HOME/jobs     # any directory on the shared home
```

**You were given a JobID (the usual case).**

1. `spur show job --controller "$SPUR" <JobID>` shows `RUNNING`; if not,
   report it and do not switch jobs.
2. Work with `spur exec --controller "$SPUR" <JobID> bash -c '...'`, using
   absolute paths (section 5).
3. Do not `scancel` it unless its owner asks.

**One task that releases the node when it ends (recommended).**

1. Put the task script and its data on the shared home (section 9).
2. Pick a node: `sinfo --controller "$SPUR" -N -o '%N|%t|%G' | grep -E "$NODE_RE"`.
   Take an idle one quickly; it can be gone within about 100 seconds.
3. Submit, without `-A`, with `-c` and `--exclusive`, and with logs on the
   shared home:

   ```bash
   sbatch --controller "$SPUR" --parsable -J "$USER-job" -p "$PART" \
     -N1 -w <node> --gpus-per-node=8 --exclusive -c 64 -t 1-00:00:00 \
     -o "$SHARED/logs/%j.out" -e "$SHARED/logs/%j.err" "$SHARED/task.sh"
   ```

4. `squeue --controller "$SPUR" -u "$USER"` shows `R`, not `PD`.
5. The job leaves the queue when it ends and the node is released. Check
   `sacct --controller "$SPUR" | grep <JobID>` for `COMPLETED` and `0:0`.

**Hold a node (you release it).**

1. Pick a node as above.
2. `sbatch --controller "$SPUR" --parsable -J "$USER-hold" -p "$PART" -N1 -w <node> --gpus-per-node=8 --exclusive -t 7-00:00:00 --wrap "sleep 604800"`
3. Work with `spur exec --controller "$SPUR" <JobID> bash -c '...'`.
4. Before it expires: `spur scontrol update --controller "$SPUR" JobId=<JobID> TimeLimit=30-00:00:00`,
   then confirm with `squeue`.
5. When done: `docker rm -f <container>`, check `rocm-smi`, then
   `scancel --controller "$SPUR" <JobID>`.

**A multi-node job you were given (old pool).**

1. `scontrol show job "$JOBID" | head -12` shows `NumNodes` and `TresPerNode`.
2. Nodes: `scontrol show job "$JOBID" | grep -oE "$NODE_RE" | sort -u`.
3. Head node: `spur exec "$JOBID" bash -lc "..."`.
4. Other nodes: `JOBID=$JOBID ~/nx.sh <node> '...'` (section 6).
5. Ranks follow node order (section 6), as observed.
6. Shared data on the shared home, never on node-local scratch.
7. Do not `scancel` it unless its owner asks.
