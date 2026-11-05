# Multi-node notes

> **Preparation only; not validated with JAX-Titan.** JAX-Titan runs on one
> node and does not support more. Nothing in this file has been run with
> JAX-Titan. Section 2 collects what the RL runbooks learned on SPUR with
> PyTorch, RCCL and Ray, on the old pool with 4 nodes of 8 MI355X on
> 2026-08-05; multi-node jobs on v2 are untested. Everything about JAX in
> sections 3 and 4 is untested.

This file keeps those lessons for the day JAX-Titan runs on several nodes.
It assumes [spur-cluster-guide.md](spur-cluster-guide.md) (Cluster facts,
reaching non-head nodes with `nx.sh`) and
[jax-titan-on-spur.md](jax-titan-on-spur.md) (container and helper scripts).

## 1. Why JAX-Titan is single-node

One controller runs the whole train step as one IFRT IR program, with XLA's
IFRT IR interpreter on the ROCm PJRT plugin. A JAX process drives only the
GPUs of its own node; GPUs on other nodes need a process of their own on each
node. An MPMD program that spans nodes therefore needs a multi-controller
IFRT IR executor, in which every process runs its part of the program and the
transfers between stage meshes cross nodes. That is milestone M5 (distributed
hardening) of the JAX-Titan build plan, listed as "Multi-node execution" in
the README's roadmap.

The no-pipeline baseline (`--parallel=dp` or `fsdp`) is one `jax.jit`
program, and JAX runs such programs across processes in general. But the
trainer never calls `jax.distributed.initialize` and feeds each batch from
its own process with `jax.device_put`. Started on several nodes as it is,
each process would train alone on its own 8 GPUs.

## 2. What the RL runs learned on SPUR

All of this section is **tested on the old pool** with PyTorch, RCCL and Ray,
unless marked otherwise.

### 2.1 Getting the nodes

The first limit is the QOS group quota of the account, not free nodes; the
RL runs used a QOS with a larger quota through `spur run -q`, with
`--gpus-per-node=8` and `--pty` (spur-cluster-guide.md section 4.5). On v2,
account-less jobs use no QOS (section 4.3 there); whether multi-node jobs
meet other limits on v2 is unknown.

### 2.2 Reaching every node

- `spur exec` reaches only the head node; `nx.sh` reaches the others through
  `srun --overlap` and `script(1)` (spur-cluster-guide.md section 6). `nx.sh`
  runs on the docker host side, so containers are reached with
  `~/nx.sh <node> 'docker exec ...'`.
- Background processes die with the `srun` step that started them:
  `nohup docker pull ... &` through `nx.sh` was killed when the step ended,
  without a message. Run long downloads in the foreground, with one subshell
  per node for parallelism, and long jobs with `docker exec -d`, which
  survived between `srun` steps.
- There is no ssh between nodes, so `rsync -e ssh` fails. To copy hundreds of
  GB between node-local disks, the RL runs started an rsync daemon in the head
  node's container (host network) and pulled from it on the other nodes in
  parallel: 805 GB in 6 minutes.
- The RL runs kept the Ray head and the driver on the head node, so that
  monitoring worked through `spur exec` without the `srun` detour.
- `pip install -e` writes into the source tree on the shared home, so the RL
  runs installed dependencies on one node after another, never in parallel.

### 2.3 Storage

- Node-local scratch is separate on each node: a 4-node job sees four
  different, empty directories. Models, data and anything every node reads go
  on the shared home.
- Reading the same files from the shared home on many nodes at once is slow.
  32 processes loading a 57 GB model took about 13 minutes, all in parallel;
  the page cache made later runs on the same nodes much faster. The
  alternative is one copy per node on node-local scratch, at the cost of the
  copies and a distribution step.
- First-run compile caches on the shared home are a risk: 32 processes would
  compile into the same directory at once. The RL runs never hit it because
  the cache already existed (so this is untested); they recommend a
  single-node run first to fill the cache.
- Multi-node checkpoints were never saved in the RL runs on SPUR. The runbook
  for another AMD cluster (3 nodes of 8 MI308X) requires the checkpoint
  directory on storage that every node sees, and says never to keep the only
  copy of a checkpoint in `/dev/shm`.

### 2.4 Containers for cross-node collectives

The single-node container needs two more things as soon as a collective spans
nodes:

- `--ulimit memlock=-1`. The host limits locked memory to 8 MB, and RDMA
  memory registration needs more. Registering 4 MB of host memory worked on
  all 9 HCAs in a container with unlimited memlock, while on the host only the
  first succeeded and the rest failed with `ENOMEM`.
- The host's ionic verbs provider, bind-mounted. Images ship libibverbs
  providers for mlx4, mlx5, efa and others, but not for the AMD Pensando ionic
  NICs, so in the container RCCL saw only the 200 Gb/s frontend NIC
  (`NET/IB : Using [0]<frontend NIC>:1/RoCE [RO]; OOB <NET_IF>:...`). With the host
  provider mounted, all 9 HCAs were visible; the host's v54 provider loaded
  fine with the image's v39 libibverbs.

From the RL container script, with the host paths from Cluster facts:

```bash
IONIC_DRIVER=<host ionic .driver file>
IONIC_LIB=$(ls <host libionic.so.* pattern> 2>/dev/null | head -1 || true)
IB_MOUNTS=()
if [ -n "$IONIC_LIB" ] && [ -f "$IONIC_DRIVER" ]; then
  IB_MOUNTS+=(-v "$IONIC_DRIVER":/etc/libibverbs.d/ionic.driver:ro)
  IB_MOUNTS+=(-v "$IONIC_LIB":/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
else
  echo "WARNING: no host ionic provider found; RCCL will fall back to TCP" >&2
fi
docker run ... --ulimit memlock=-1 "${IB_MOUNTS[@]}" ...
```

The RL containers also ran with `--network=host`, `--ipc=host`,
`--privileged`, `--shm-size 64G` and `--ulimit nofile=1048576:1048576`; with
the default soft limit of 1024 open files, Ray's raylet failed with
`Too many open files`.

### 2.5 Network interfaces

- `$NET_IF` (Cluster facts) is the only interface that routes between nodes;
  it carries TCP at 200 Gb/s. With `--network=host`, containers also see
  `docker0` and `br-*` addresses (172.x), which do not route between nodes, so
  RCCL, Gloo and Ray had to be kept off them:
  `NCCL_SOCKET_IFNAME=$NET_IF`, `GLOO_SOCKET_IFNAME=$NET_IF`, and Ray's
  `--node-ip-address` set to the `$NET_IF` address.
- A node's `$NET_IF` address:

  ```bash
  ip -o -4 addr show "$NET_IF" | awk '{print $4}' | cut -d/ -f1
  ```

- RCCL reads NCCL's variable names (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`,
  `NCCL_IB_GID_INDEX`, `NCCL_IB_DISABLE`, `NCCL_DEBUG`).
- On another AMD cluster (MI308X), a leftover `NCCL_SOCKET_IFNAME` that named
  an interface the node did not have made RCCL fail with
  `Bootstrap: no socket interface found`, which surfaced as `ncclGetUniqueId`
  or `NCCL error: invalid usage`.

### 2.6 RDMA does not work from containers yet

Cross-node collectives ran over TCP on `$NET_IF`, with `NCCL_IB_DISABLE=1`. A
32-rank all-reduce across 4 nodes passed and the RL smoke runs finished.

Each node has 9 active HCAs: 8 ionic NICs at 400 Gb/s for the GPU network and
the 200 Gb/s frontend NIC (Cluster facts). To list them with their state and
rate:

```bash
for d in /sys/class/infiniband/*; do n=$(basename $d); \
  printf "%-10s %-12s %s\n" "$n" "$(cat $d/ports/1/state)" "$(cat $d/ports/1/rate)"; done
```

The ionic NICs have no IPv4 address, only a routable IPv6 address each, and
the same NIC sits in a different /64 on every node: a routed RoCE fabric, so
the GID index matters. They were not usable from the containers. Of four
problems, three were solved:

1. The container did not see the ionic devices: mount the host provider
   (section 2.4).
2. memlock was capped at 8 MB: `--ulimit memlock=-1`.
3. GID index: on the ionic NICs, `gid[0]` is link-local (`fe80::`) and
   `gid[1]` the routable address. With the default index 0, RCCL
   hung: the channels never came up, and it timed out after 90 seconds.
   `NCCL_IB_GID_INDEX=1` is required.
4. **Unsolved:** with the first three fixed, queue pairs came up, but
   registering RCCL's proxy buffers failed:

   ```
   NCCL WARN Call to ibv_reg_mr failed with error Invalid argument
   ```

   Ruled out, each by a measurement: GPU Direct RDMA (already disabled for
   the GPU/NIC distance; RCCL used host staging buffers), PCI relaxed ordering
   (`NCCL_IB_PCI_RELAXED_ORDERING=0`), dmabuf and GDR settings
   (`NCCL_DMABUF_ENABLE=0`, `NCCL_NET_GDR_LEVEL=0`), a single bad NIC (the
   frontend NIC alone, one ionic NIC alone and `NCCL_IB_MERGE_NICS=0` all
   failed the same way), pinned host memory (a ctypes probe registered
   `malloc` and `hipHostMalloc` memory fine; only GPU memory failed, with
   `EFAULT` rather than `EINVAL`), and the rdma-core version mix.

The likely cause is that RCCL's built-in IB transport does not suit these
AINIC NICs. The cluster's RCCL profile (Cluster facts) selects AMD's ANP
network plugin and sets AMD's recommended values:

```bash
export NCCL_NET_PLUGIN=anp
export NCCL_DMABUF_ENABLE=1
export NCCL_IB_GID_INDEX=1
export NCCL_IB_TC=96
export NCCL_IB_FIFO_TC=184
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_USE_INLINE=1
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_GDRCOPY_ENABLE=0
export IONIC_LOCKFREE=all
```

RCCL's `NET/Plugin: Could not find: librccl-net.so. Using internal net plugin.`
says as much. The plugin did not load in the RL containers: it needs
`GLIBC_2.38` (three symbols, `__isoc23_fscanf`, `__isoc23_strtol` and
`__isoc23_strtoll`), and their image was Ubuntu 22.04 with glibc 2.35. The
hosts run Ubuntu 24.04, and the AINIC packages exist only for 24.04; mounting
the host's libibverbs v54 hit `GLIBC_2.38` too. The RL runbook's next steps,
in order: inject the host's rdma-core and plugin dependencies with a glibc
2.38+ loader (possible but fragile), rebuild on a base image with glibc 2.38
or newer (costly but clean), or get a plugin built for Ubuntu 22.04.

**For JAX-Titan (untested):** its image is based on Ubuntu 24.04
(`jax-base-ubu24`), so the glibc requirement may already be met. Whether the
host's plugin, which sits next to the host's ROCm 7.0.1, works with the RCCL
of TheRock 10.1 in the container is unknown. Try it only after a TCP run
works.

To see which transport RCCL picked, run with `NCCL_DEBUG=INFO`. Two RL
runbooks for other AMD clusters with MI308X nodes, where GPU Direct RDMA
worked, accept a run only if the log shows `NCCL INFO Using network IB` and
`NET/IB/<n>/GDRDMA`; socket lines alone mean TCP. Both mapped the host's
`/dev/infiniband` into their containers and set `NCCL_IB_HCA`,
`NCCL_IB_GID_INDEX` and `NCCL_SOCKET_IFNAME` to the NIC, GID index and
interface of that cluster (the three-node one used an ionic NIC with GID
index 1). The two-node runbook also set `--ulimit memlock=-1`; the three-node
one ran its containers `--privileged`, and without that RCCL fell back to
sockets. The two-node runbook discovers the HCA, interface and GID index from
sysfs, accepting only RoCE v2 GIDs on interfaces with an IPv4 address; on
SPUR, where the ionic NICs have none, that discovery would find no candidate
(not tried).

### 2.7 Ranks and nodes

Ray handed out ranks contiguously by node (spur-cluster-guide.md section 6):
the head node had ranks 0-7, the next node 8-15, and so on. This was
observed, not guaranteed, so the RL code checked it wherever a group had to
stay inside one node. It decided placement: with Megatron's rank order and
TP = CP = 1, expert parallelism of 8 put each expert-parallel group on one
node, which kept the MoE all-to-all on the intra-node links and left only
gradient reductions for the network. TP=4 with EP=8 would have spread every
expert-parallel group over all four nodes, a bad trade while inter-node
traffic runs over TCP.

### 2.8 The environment must reach every node

Ray does not pass the driver's environment to workers on other nodes; they
inherit the environment their node's raylet started with. Every variable had
to be set on every node before `ray start`, or the workers came up without
it, with no error, only drifting metrics. Setting RCCL variables on the
driver had no effect: one probe "passed" with `NCCL_IB_DISABLE=1` exported on
the driver, while the workers never received it. Check the environment
inside each process, not where you launched it.

### 2.9 Cheap checks before a long run

A smoke run took more than ten minutes to its first metrics, so the RL runs
first ran two probes: one that reported, for every node, whether the code
imported and which switches were set, and an all-reduce over all 32 ranks
that checked the placement (8 per node) and the result. Section 3.6 sketches
the JAX equivalent of the second.

### 2.10 Ports on the host network

With `--network=host`, every container shares the node's port space. When a
second Ray cluster was started on nodes that already ran one, a single port
left at its default, the dashboard agent's, collided. The agent registered
its port as `None`, and job submission failed with
`Invalid URL: port can't be converted to integer`, far from the cause. Every
port of the second cluster had to be moved. Also, starting the Ray head and
joining a worker in one command made the worker hang on a head that was not
up yet; start the head, check it, then join.

## 3. Mapping to JAX multi-process training

**Everything in this section is untested.**

### 3.1 Processes and the coordinator

- Run one JAX process per node, owning that node's 8 GPUs. Put process 0 on
  the head node, which `spur exec` reaches directly.
- Every process calls `jax.distributed.initialize` before any other JAX call,
  with the coordinator's address, the number of processes and its own index:

  ```python
  jax.distributed.initialize(
      coordinator_address=f"{head_ip}:{port}",
      num_processes=num_nodes,
      process_id=node_index,
  )
  ```

- The coordinator address is the head node's `$NET_IF` address (section 2.5)
  and a port that is free on the head node; containers share the host's ports
  (section 2.10), so check with `ss -ltn` first. JAX's coordination service
  connects to that address, so it must not be a `docker0` or `br-*` address.
- Pass the arguments explicitly. JAX can detect launches under Slurm from
  `SLURM_*` variables, but here the processes start inside docker containers,
  not as `srun` tasks.
- All processes must reach the coordinator within `initialization_timeout`
  (300 seconds by default), so start the non-head processes right after
  process 0.
- Afterwards, each process should see `jax.process_count()` equal to the
  number of nodes, 8 `jax.local_devices()` and 8 times as many
  `jax.devices()`.

### 3.2 Environment per process

- No driver hands out the environment (section 2.8): set the RCCL variables in
  each process's own environment, inside each node's container, and print
  them at startup. Start with TCP:

  ```bash
  export NCCL_SOCKET_IFNAME=$NET_IF NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO
  ```

- `GLOO_SOCKET_IFNAME` is a PyTorch setting; JAX does not use Gloo.
- Use the same image digest, wheels, venv and `XLA_FLAGS` on every node. The
  wheels and venv on the shared home make this the default.

### 3.3 Devices, meshes and stages

- `jax.devices()` is expected to list each process's devices together, but
  check `[(d.id, d.process_index) for d in jax.devices()]` before relying on
  the order, as the RL runs checked their ranks.
- JAX-Titan's `make_topology` gives each stage a block of consecutive devices.
  With 8 GPUs per stage and devices grouped by process, each stage mesh would
  be one node: data parallelism, FSDP and tensor parallelism inside a stage
  would stay on intra-node links, and only the transfers between stages would
  cross the network. This is the same reasoning as expert parallelism of 8 in
  section 2.7. Tensor-parallel groups, also consecutive devices, stay inside a
  node when their size divides 8.

### 3.4 Data and caches

- The token file goes on the shared home, where every node sees it.
- The JAX compilation cache in `$JT_ROOT/cache/jax` is shared too. MPMD
  programs do not use it yet, but the no-pipeline programs do:
  fill it with a single-node run first, or give each process its own
  directory (section 2.3).

### 3.5 Starting the processes

1. Hold the nodes as one multi-node job (spur-cluster-guide.md section 4.5,
   old pool).
2. Check each node: `rocm-smi`, and `scripts/gpus_free.sh` outside the
   container.
3. Start the container on each node with `container.sh`: through `on_node`
   on the head node, through `nx.sh` on the others. For RDMA, add the flags of
   section 2.4 to it.
4. Start process 0 on the head node, then the others, each detached with
   `jt.sh -d`.
5. Read every process's log on the shared home.

### 3.6 Probe: an all-reduce over every GPU of every node

A minimal check that RCCL connects all GPUs across nodes. It says nothing
about MPMD. Write the probe next to the helper scripts of
jax-titan-on-spur.md:

```bash
cat > "$JT_SPUR/probe_allreduce.py" <<'EOF'
"""Sums one value per GPU over every GPU of every node.

  python probe_allreduce.py PROCESS_ID NUM_PROCESSES COORDINATOR_HOST:PORT
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

process_id, num_processes, coordinator = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
jax.distributed.initialize(
    coordinator_address=coordinator,
    num_processes=num_processes,
    process_id=process_id,
)
mesh = jax.make_mesh((jax.device_count(),), ("d",))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
x = jax.make_array_from_process_local_data(
    sharding, np.ones(jax.local_device_count(), np.float32)
)
total = jax.jit(jnp.sum)(x)
print(
    f"process {jax.process_index()}/{jax.process_count()}: "
    f"{jax.local_device_count()} local, {jax.device_count()} global devices, "
    f"sum {float(total.addressable_data(0))}",
    flush=True,
)
EOF

cat > "$JT_SPUR/probe.sh" <<'EOF'
#!/usr/bin/env bash
# Runs probe_allreduce.py over TCP on the given interface; logs to
# $JAXTITAN_ROOT/logs/probe_PROCESS_ID.out.
#   probe.sh PROCESS_ID NUM_PROCESSES COORDINATOR_HOST:PORT INTERFACE
set -euo pipefail
source scripts/env.sh
exec > "$JAXTITAN_ROOT/logs/probe_$1.out" 2>&1
export NCCL_SOCKET_IFNAME=$4 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO
env | grep '^NCCL_' || true
python "$(dirname "$(readlink -f "$0")")/probe_allreduce.py" "$1" "$2" "$3"
EOF
```

Launch it from the login node, with the containers running on every node:

```bash
NET_IF=<routable inter-node interface>      # Cluster facts
NODES=(<head node> <second node>)           # the job's nodes, head first
N=${#NODES[@]}
HEAD_IP=$(on_node "ip -o -4 addr show $NET_IF | awk '{print \$4}' | cut -d/ -f1")
COORD=$HEAD_IP:<port that is free on the head node>

on_node "$JT_SPUR/jt.sh -d bash $JT_SPUR/probe.sh 0 $N $COORD $NET_IF"
for i in $(seq 1 $((N - 1))); do
  JOBID=$JOBID ~/nx.sh "${NODES[$i]}" \
    "$JT_SPUR/jt.sh -d bash $JT_SPUR/probe.sh $i $N $COORD $NET_IF"
done
sleep 60; grep -h '^process ' "$JT_ROOT"/logs/probe_*.out
```

`on_node` passes `--controller "$SPUR"`; on the old pool, where multi-node
jobs were tested, set `SPUR` to the old pool's controller. Every process
should print `8 local`, `8N global devices` and a sum of `8N`. The
`NCCL INFO` lines in the same logs show the transport (section 2.6).

## 4. Open questions for the first multi-node JAX-Titan run

- A multi-controller IFRT IR executor (milestone M5). Until it exists,
  JAX-Titan's pipelines cannot span nodes.
- Reaching non-head nodes on v2: whether `spur exec` is head-only there and
  whether `nx.sh` works with `srun --controller`.
- Multi-node allocation limits on v2 for account-less jobs.
- RCCL RDMA from the JAX-Titan container (section 2.6).
- Saving and restoring checkpoints from several processes on the shared home.
