# AMD cluster runbooks for JAX-Titan

These runbooks explain how to run JAX-Titan on SPUR, AMD's internal cluster of
nodes with 8 AMD Instinct MI355X (gfx950) GPUs each. They are written for
people and for coding agents that work on the cluster on someone's behalf.

They are adapted from an internal collection of Chinese runbooks for RL
training (LumenRL, DAPO, Megatron, vLLM, Ray and others). Only the lessons
about the cluster, containers, storage, networking and debugging carry over;
[source-runbooks.md](source-runbooks.md) lists every source file and what was
kept. For JAX-Titan itself, [getting_started.md](../getting_started.md)
remains the reference: these files wrap its commands and do not repeat its
measured results.

> **Review before publishing.** These files describe an AMD-internal cluster.
> Every cluster-specific value (controller URLs, login and compute node names,
> partitions, accounts and QOS, shared and scratch paths, NIC names, host file
> paths) is in the Cluster facts table at the top of
> [spur-cluster-guide.md](spur-cluster-guide.md), and the other files refer to
> them through shell variables. The files also describe internal scheduler
> behavior, quotas, an example team node pool and the cluster's RDMA setup.
> Review or remove them before this directory is published with the
> repository.

## Files

| File | Contents |
|---|---|
| [spur-cluster-guide.md](spur-cluster-guide.md) | Cluster facts; rules for agents; connecting, allocating in task or hold mode, running commands on nodes, reaching every node of a multi-node job, renewing, releasing; storage, reliability, troubleshooting |
| [jax-titan-on-spur.md](jax-titan-on-spur.md) | JAX-Titan on one node: container, wheels, venv, data, CPU check, training, logs, task mode, cleanup and release, checklist |
| [multi-node-notes.md](multi-node-notes.md) | What the multi-node RL runs learned, and how it maps to JAX multi-process training; preparation only |
| [source-runbooks.md](source-runbooks.md) | Every source runbook, with what carries over |

## For agents

Read the rules for agents (section 1) and the Cluster facts of
spur-cluster-guide.md before running any command. In short: use only the
JobID you were given; do not allocate nodes unless asked; never ssh to compute
nodes; never `scancel` a job that is not yours; release what you allocate.
Then follow jax-titan-on-spur.md, whose checklist gives the order of work.
After every run, `scripts/cleanup_after_train.sh` frees the GPUs for the next
user. When a step marked untested works or fails, record the result in the
file.

## What is tested

| Topic | Status |
|---|---|
| SPUR v2: connecting, task and hold mode, `spur exec`, renewing, releasing, storage | Tested on v2 on 2026-09-16 (source runbooks) |
| SPUR old pool: interactive allocation, quotas, multi-node allocation, `nx.sh`, rank placement, multi-node storage and container flags, TCP between nodes | Tested on the old pool with PyTorch, RCCL and Ray; multi-node with 4 nodes of 8 GPUs on 2026-08-05 |
| RDMA over the ionic NICs from containers | Does not work yet: one unsolved error |
| JAX-Titan: build, venv, data, CPU check, training | Tested on a shared 8x MI355X node outside SPUR (getting_started.md) |
| `scripts/cleanup_after_train.sh` | Dry run tested on that node; `--yes` untested |
| JAX-Titan on a SPUR node: the adapted container script, helper scripts, storage layout, task-mode script | Untested |
| Multi-node jobs on v2 | Untested |
| JAX-Titan on more than one node | Not supported; multi-node-notes.md is preparation |

Each file marks what it took from tested runs, and on which pool, and marks
everything else, including all JAX-specific steps derived for SPUR, as
untested.
