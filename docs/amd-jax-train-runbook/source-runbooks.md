# Source runbooks

The files in this directory are adapted from an internal collection of
Chinese runbooks and handoff notes for RL training on AMD GPUs (LumenRL, DAPO,
Megatron, vLLM, SGLang, Ray, Miles, Vime, DeepSeek-V4, Qwen3). Most of them
were written on SPUR; a few on other AMD clusters with MI300X or MI308X nodes.
This table lists every file of the collection and what, if anything, carries
over to JAX-Titan.

| Source file | What it is | Carries over |
|---|---|---|
| `SPUR_NODE_ACCESS_GUIDE.md` | How agents and people reach SPUR jobs and nodes on the old pool: agent rules, accounts and QOS quotas, interactive allocation, `spur exec`, multi-node access with `nx.sh`, troubleshooting | Yes: [spur-cluster-guide.md](spur-cluster-guide.md) |
| `spur-v2-cluster-guide.md` | The v2 pool, tested on 2026-09-16: `--controller`, task and hold mode, running, renewing, releasing, storage, reliability | Yes: [spur-cluster-guide.md](spur-cluster-guide.md) |
| `dapo-lumenrl-4node-32gpu-runbook.md` | LumenRL DAPO on 4 nodes of 8 MI355X: quotas, `nx.sh`, shared storage, container flags for RDMA, the failed ionic RDMA investigation and the TCP fallback, Ray setup, smoke results | Multi-node mechanics: [multi-node-notes.md](multi-node-notes.md). Ray and RL parts dropped |
| `lumenrl-megatron-native-env-build-runbook.md` | Building the LumenRL Megatron-native stack (PyTorch, vLLM, Megatron-Core, ROCm TransformerEngine and Apex from source) on a new gfx950 node | General practice only: pin the image digest, keep large build trees off the shared home, in-container installs are lost with `docker rm` |
| `rocm-gfx950-strided-bmm-oob-issue.md` | `torch.bmm` reads out of bounds on gfx950 with ROCm 7.0 for interleaved batch views; fixed with `.contiguous()` | Debugging lessons: GPU core dumps of 50-80 GB land in the working directory; after a fault, test one case per process. The bug itself is PyTorch-specific |
| `deepseek-v4-miles-megatron-4node-runbook.md` | Miles Megatron RL for DeepSeek-V4-Flash on 4 SPUR nodes | Cluster lessons: background processes die with `srun` steps, no ssh between nodes (rsync daemon), port collisions on the host network, the container's ROCm user space is independent of the host's |
| `qwen3-30b-a3b-lumenrl-megatron-runbook.md` | Two-node Qwen3-30B-A3B RL on MI308X (not SPUR), with RCCL GPU Direct RDMA weight sync | RDMA checks: `/dev/infiniband` and memlock in the container, sysfs discovery of NIC and GID index, RCCL log lines that prove RDMA |
| `deepseek-v4-flash-lumenrl-megatron-runbook.md` | Three-node DeepSeek-V4-Flash RL on MI308X (not SPUR), with RCCL RDMA weight sync | RDMA and interface pitfalls (`--privileged`, a wrong `NCCL_SOCKET_IFNAME`), checkpoints on shared storage, `/dev/shm` staging |
| `megatron-moe-length-collapse-handoff.md` | Investigation of Megatron response-length collapse for Qwen3-30B-A3B on a SPUR node | Usage pattern only: `spur exec ... docker exec -d` for long runs; node-local data is rebuilt after a node change |
| `dapo-lumenrl-vime-fsdp-megatron-runbook.md` | LumenRL DAPO on the `vllm/vime-rocm` image with FSDP2 and Megatron-native backends, including MoE | Two engines cannot share the same GPUs (memory budgets are fractions of the whole GPU); the idle memory baseline |
| `dapo-miles-fsdp-sglang-runbook.md` | Miles DAPO with FSDP2 and SGLang, matched to LumenRL for comparison | `--ulimit nofile` in containers that run Ray (multi-node notes) |
| `dapo-lumenrl-native-vllm-fsdp-runbook.md` | LumenRL-native DAPO in BF16 and FP8 on one 8-GPU node: FSDP2 training with colocated vLLM rollout | No |
| `dapo-lumenrl-megatron-bf16-new-machine-runbook.md` | New-machine runbook for LumenRL Megatron-native BF16 DAPO with Qwen3-30B-A3B | No |
| `dapo-lumenrl-vllm-fsdp-megatron-new-machine-runbook.md` | New-machine runbook for both LumenRL backends (FSDP2 DP8, Megatron-native DP2xTP2xPP2) with Qwen3-8B | No |
| `dapo-vime-vllm-megatron-runbook.md` | Vime DAPO with Megatron training and vLLM rollout on one node | No |
| `dapo-lumenrl-megatron-moe-ep-handoff.md` | Handoff for continuing MoE and expert-parallel work in LumenRL on a new machine | No |
| `dapo-lumenrl-moe-fsdp-vllm-handoff.md` | Handoff on train/rollout consistency for Qwen3 MoE with FSDP2 and vLLM (expert weights lost in weight sync) | No |
| `lumenrl-rollout-tp-gt-1-handoff.md` | Colocated vLLM rollout with tensor parallelism above 1 in LumenRL | No |
| `megatron-native-refactor-handoff.md` | Refactoring LumenRL's Megatron backend onto Megatron-Core (TP, PP, CP, distributed checkpoints) | No |
| `atom_vllm_rollout_compare_runbook.md` | vLLM against ATOM long-decode rollout benchmark, and an online weight-update bug reproduction | No |
| `deepseek-v4-base-rl-train-handoff.md` | Handoff for DeepSeek-V4-Flash-Base DAPO training on LumenRL | No (its note on shared-home space is in Cluster facts) |
| `deepseek-v4-flash-enablement-handoff.md` | How far DeepSeek-V4-Flash gets on LumenRL and ROCm | No |
| `deepseek-v4-lumenrl-fsdp-handoff.md` | Handoff for DeepSeek-V4 RL on LumenRL with FSDP2 and vLLM | No |
| `deepseek-v4-lumenrl-megatron-handoff.md` | Handoff for DeepSeek-V4 RL on LumenRL's Megatron-native backend | No |
| `dsv4-flash-vllm-inference-runbook.md` | DeepSeek-V4-Flash vLLM inference on one 8x MI300X node, BF16 with online FP8 | No |
| `kimi-k3-dspark-sddd-lumenrl-runbook.md` | Kimi K3 DSpark draft-model distillation with LumenRL and vLLM on 8x MI350 | No |
| `agent-dev/README.md` | SFT data collection for a kernel-optimization agent | No |
| `agent-dev/kernel-agent-sft-data-design.md` | Design of the SFT data for a kernel analysis agent | No |
| `agent-dev/qwen3-30b-a3b-agentic-rl-geak-runbook.md` | Agentic RL for a kernel agent on 2 SPUR nodes (MI355X and MI300X); records no results | Restates the agent rules; nothing new |
| `agent-dev/skill-offline-sft-extraction.md` | Agent skill: extract SFT samples from saved agent trajectories | No |
| `agent-dev/skill-online-sft-collection.md` | Agent skill: record agent events for SFT with a buffer-only middleware | No |
| `agent-dev/sft_collector/` | Python package behind the two skills | No |

## What carried over

- Everything about the cluster itself: connecting, allocating, entering
  nodes, renewing, releasing, storage, reliability and the agent rules, with
  their tested and untested markers.
- Multi-node mechanics: reaching non-head nodes, storage across nodes,
  container flags, interface choice, the RDMA investigation, rank placement,
  environment propagation and cheap pre-run probes. JAX-Titan cannot use them
  yet, so they are kept as preparation.
- Debugging lessons that apply to any ROCm job: GPU core dumps on the shared
  home, leftover containers holding GPU memory, one GPU job at a time.

## What was dropped, and why

- RL procedures (rollout engines, weight sync, DAPO and GRPO recipes and
  metrics, Ray cluster setup and its shims): JAX-Titan has no RL training
  yet, and none of it maps onto a JAX training step.
- Framework environment builds (Megatron, TransformerEngine, Apex, vLLM,
  aiter): JAX-Titan builds its own stack with `scripts/build_jax_rocm.sh`.
- Model-specific bugs and handoffs (DeepSeek-V4, Qwen3 MoE, Kimi K3): they
  concern PyTorch code paths that JAX-Titan does not use.
- Personal user names, e-mail addresses, job IDs, IP addresses and host names
  of other clusters: replaced by `$USER`, `<JobID>` and placeholders, or left
  out.
