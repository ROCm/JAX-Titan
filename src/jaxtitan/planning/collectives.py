"""Collective communication by message size, for RCCL and MORI.

  python -m jaxtitan.planning.collectives --op=all_reduce --ranks=8 --gpu=MI355X
  python -m jaxtitan.planning.collectives --op=all_reduce --ranks=1024 --nodes=128
  python -m jaxtitan.planning.collectives --op=dispatch --ranks=16 --nodes=2 --tokens=4096
  python -m jaxtitan.planning.collectives --calibrate logs/collectives/2026-11-08 --json table.json
  python -m jaxtitan.planning.collectives --op=all_gather --ranks=8 --table table.json

A collective has a library, an operation, a group of `ranks` over `nodes`, and a message
size S. It takes a fixed latency plus its bytes on the wire over the bandwidth of its
slowest path, for the algorithm the library picks at that size. Sizes and bus bandwidth
follow rccl-tests: S is the buffer a rank contributes or receives in full, and bus
bandwidth is S / time times the operation's factor, so it compares with link bandwidth.

RCCL:
- all-reduce: ring, 2(n - 1) steps carrying 2(n - 1)/n S per rank; or a tree, with
  latency growing as log2 of the nodes instead of n, RCCL's pick for small messages.
- all-gather and reduce-scatter: ring, (n - 1) steps carrying (n - 1)/n S.
- all-to-all: direct, (n - 1)/n S per rank, the part for other nodes over the NICs.
- send/recv: S over one link.
Each runs with RCCL's LL protocol (low latency, 29% of the bandwidth) or Simple (all of
it after filling a pipeline on each step), whichever is faster at that size. Inside a
node the bandwidth is the links among the group's GPUs (k - 1 links of a fully
connected platform for k GPUs); across nodes a ring is also bounded by the group's k
NICs per node, and pays the inter-node hop only on the steps that cross nodes.

MORI-EP dispatch and combine: each rank sends each token to the ranks of its top-k
experts, tokens x k x hidden x bytes per element, the share for other nodes over RDMA
and the rest over XGMI. The defaults are MORI's published bandwidth and latency
(README at 752efd37: MI300X with ConnectX-7, MI355X with AINIC).

Links reach `efficiency` of their peak and RCCL adds a fixed latency per collective;
measured tables (`--table`, from `--calibrate` of scripts/bench_collectives.sh outputs,
the RCCL Inspector or MORI's perf records) replace the model where they cover the
group and size.
"""

import argparse
import dataclasses
import functools
import json
import math
import pathlib
import re

from jaxtitan.planning import gpus as gpus_lib

OPS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "send_recv",
       "dispatch", "combine")
# Calibrated with rccl-tests 2.18.3 and RCCL 2.30.7 on 8x MI355X, one thread per GPU as
# JAX drives them (docs/planning.md): collectives reach 74-80% of the links' peak; a ring
# all-reduce takes 7.5 us plus about 1 us per step inside a node (9.4, 14.1 and 22.4 us
# for 2, 4 and 8 GPUs); a forced tree all-reduce reaches a third of the ring's bandwidth.
# Across nodes, the per-step latency and the NICs' efficiency are assumptions until a
# multi-node sweep measures them.
EFFICIENCY = 0.75
NIC_EFFICIENCY = 0.9
RCCL_LATENCY_US = 7.5
HOP_US = {"intra": 1.0, "inter": 5.0}
TREE_BANDWIDTH = 0.35
# RCCL's protocols: LL has the latency above at 29% of the bandwidth; Simple has all of
# it after filling a pipeline of chunks on each step (8-GPU all-reduce: 38 us at 1 MiB
# with LL, 135 us at 16 MiB with Simple).
LL_BANDWIDTH = 0.29
SIMPLE_STEP_US = 2.0
DATA = pathlib.Path(__file__).parent / "data"

_RCCL_NAMES = {"all_reduce": "all_reduce", "allreduce": "all_reduce",
               "all_gather": "all_gather", "allgather": "all_gather",
               "reduce_scatter": "reduce_scatter", "reducescatter": "reduce_scatter",
               "alltoall": "all_to_all", "all_to_all": "all_to_all",
               "sendrecv": "send_recv", "send_recv": "send_recv", "send": "send_recv",
               "recv": "send_recv", "broadcast": "broadcast", "reduce": "reduce"}

# MORI-EP on DeepSeek V3 shapes (hidden 7,168, top-8, FP8 dispatch, BF16 combine):
# bandwidth with 4,096 tokens per rank over XGMI and RDMA, latency with 128 tokens.
MORI_EP_PUBLISHED = {
    "AMD Instinct MI300X": {
        8: {"dispatch": (307, None, 35), "combine": (330, None, 47)},
        16: {"dispatch": (171, 52, 76), "combine": (219, 67, 122)},
        32: {"dispatch": (103, 57, 157), "combine": (91, 50, 280)},
    },
    "AMD Instinct MI355X": {
        8: {"dispatch": (345, None, 31), "combine": (420, None, 36)},
        16: {"dispatch": (179, 54, 84), "combine": (234, 71, 108)},
        32: {"dispatch": (85, 46, 152), "combine": (110, 61, 187)},
    },
}
MORI_SAME_PLATFORM = {"AMD Instinct MI325X": "AMD Instinct MI300X",
                      "AMD Instinct MI350X": "AMD Instinct MI355X"}
MORI_SOURCE = "MORI README at 752efd37"


def bus_factor(op: str, n: int) -> float:
  """rccl-tests' bus bandwidth over algorithm bandwidth."""
  if op == "all_reduce":
    return 2 * (n - 1) / n
  if op in ("all_gather", "reduce_scatter", "all_to_all"):
    return (n - 1) / n
  return 1.0


@dataclasses.dataclass
class Point:
  """One measured collective: rccl-tests, the RCCL Inspector or MORI."""
  library: str
  op: str
  ranks: int
  nodes: int
  bytes: float
  time_us: float
  busbw_gbs: float | None = None
  algorithm: str | None = None
  protocol: str | None = None
  channels: int | None = None
  forced: str | None = None
  gpu: str | None = None
  source: str = ""


def _scaleout(gpu: gpus_lib.GPU, scaleout_gbs: float | None) -> float:
  bw = scaleout_gbs or gpu.scaleout_gbs
  if not bw:
    raise ValueError(f"{gpu.name} has no scale-out bandwidth; pass scaleout_gbs")
  return bw


def _bandwidth(gpu, k: int, nodes: int, efficiency: float, scaleout_gbs) -> float:
  """Per-rank bus bandwidth of a ring over k GPUs per node on `nodes` nodes."""
  inside = gpu.scaleup_gbs_for(k) * efficiency if k > 1 else math.inf
  if nodes == 1:
    return inside
  return min(inside, k * _scaleout(gpu, scaleout_gbs) * NIC_EFFICIENCY)


def _mori(op, nbytes, ranks, nodes, gpu):
  """MORI-EP from the published points: a fixed latency plus the slower of the XGMI and
  RDMA shares."""
  name = MORI_SAME_PLATFORM.get(gpu.name, gpu.name)
  table = MORI_EP_PUBLISHED.get(name)
  if not table:
    return None
  ep = next((e for e in sorted(table) if e >= ranks), max(table))
  xgmi, rdma, latency_us = table[ep][op]
  if nodes > 1 and not rdma:
    rdma = table[max(e for e in table if table[e][op][1])][op][1]

  def transfer(b, group, group_nodes):
    local = b * (group // group_nodes) / group
    return max(local / (xgmi * 1e3), (b - local) / (rdma * 1e3) if group_nodes > 1 else 0.0)

  # The fixed part: the published latency less the transfer of its 128 tokens.
  reference = 128 * 7168 * 8 * (1 if op == "dispatch" else 2)
  latency = max(0.0, latency_us - transfer(reference, ep, max(1, ep // 8)))
  moved = transfer(nbytes, ranks, nodes)
  return {"latency_us": latency, "transfer_us": moved,
          "algorithm": f"mori-ep{ep}",
          "source": MORI_SOURCE + (f" ({name})" if name != gpu.name else "")
                    + (f", EP{ep} values" if ep != ranks else "")}


def model(op: str, nbytes: float, ranks: int, nodes: int = 1, *, gpu: gpus_lib.GPU,
          library: str = "rccl", algorithm: str = "auto", efficiency: float = EFFICIENCY,
          scaleout_gbs: float | None = None) -> dict:
  """Time of one collective from the link model."""
  if ranks < 2:
    return {"latency_us": 0.0, "transfer_us": 0.0, "time_us": 0.0, "busbw_gbs": 0.0,
            "algorithm": "none", "bound": "none", "source": "model"}
  if ranks % nodes:
    raise ValueError(f"{ranks} ranks do not split evenly over {nodes} nodes")
  k = ranks // nodes
  if op in ("dispatch", "combine") and library == "mori":
    out = _mori(op, nbytes, ranks, nodes, gpu)
    if out:
      return _result(out, bus_factor("all_to_all", ranks) * nbytes)
    library = "rccl"  # no MORI numbers for this GPU: RCCL's all-to-all
  if op in ("dispatch", "combine"):
    op = "all_to_all"
  if library != "rccl":
    raise ValueError(f"MORI is modeled for EP dispatch and combine, not {op}")
  wire = bus_factor(op, ranks) * nbytes
  candidates = {}  # algorithm/protocol: (latency, transfer) in microseconds

  def add(name, intra_steps, inter_steps, transfer):
    # As NCCL's tuning model counts it: a chunk's path crosses nodes on `inter_steps` of
    # its steps. Inside a node every collective costs about a ring all-reduce's steps.
    start = RCCL_LATENCY_US + intra_steps * HOP_US["intra"] + inter_steps * HOP_US["inter"]
    candidates[f"{name}/ll"] = (start, transfer / LL_BANDWIDTH)
    candidates[f"{name}/simple"] = (start + (intra_steps + inter_steps) * SIMPLE_STEP_US,
                                    transfer)

  if op in ("all_reduce", "all_gather", "reduce_scatter"):
    passes = 2 if op == "all_reduce" else 1
    inter = passes * (nodes - 1)
    bw = _bandwidth(gpu, k, nodes, efficiency, scaleout_gbs) * 1e3
    add("ring", max(passes * (ranks - 1) - inter, 2 * (k - 1)), inter, wire / bw)
    if op == "all_reduce":
      add("tree", 2 * (k - 1), 2 * math.ceil(math.log2(nodes)), wire / (bw * TREE_BANDWIDTH))
  elif op == "all_to_all":
    inside = gpu.scaleup_gbs_for(k) * efficiency * 1e3 if k > 1 else math.inf
    local, remote = nbytes * (k - 1) / ranks, nbytes * (ranks - k) / ranks
    nic = _scaleout(gpu, scaleout_gbs) * NIC_EFFICIENCY * 1e3 if remote else math.inf
    add("direct", 2 * (k - 1), 1 if nodes > 1 else 0, max(local / inside, remote / nic))
  else:  # send/recv, broadcast, reduce: one link, or one NIC across nodes
    link = (_scaleout(gpu, scaleout_gbs) * NIC_EFFICIENCY if nodes > 1
            else gpu.scaleup_gbs_for(2) * efficiency) * 1e3
    add("p2p", 0 if nodes > 1 else 2, 2 if nodes > 1 else 0, wire / link)
  if algorithm != "auto":
    candidates = {name: v for name, v in candidates.items() if name.split("/")[0] == algorithm}
  name, (latency, transfer) = min(candidates.items(), key=lambda kv: sum(kv[1]))
  return _result({"latency_us": latency, "transfer_us": transfer, "algorithm": name,
                  "source": "model"}, wire)


def _result(parts: dict, wire: float) -> dict:
  time = parts["latency_us"] + parts["transfer_us"]
  return {**parts, "time_us": time, "busbw_gbs": wire / time / 1e3 if time else 0.0,
          "bound": "latency" if parts["latency_us"] > parts["transfer_us"] else "bandwidth"}


class Table:
  """Measured points, interpolated by message size within a group."""

  def __init__(self, points: list[Point]):
    self.points = points

  @classmethod
  def load(cls, path) -> "Table":
    return cls([Point(**p) for p in json.loads(pathlib.Path(path).read_text())["points"]])

  def save(self, path, compact: bool = False) -> None:
    data = {"points": [dataclasses.asdict(p) for p in self.points]}
    if not compact:
      data["groups"] = groups(self.points)
    pathlib.Path(path).write_text(json.dumps(data, indent=None if compact else 1) + "\n")

  def lookup(self, library, op, ranks, nodes, nbytes, gpu=None, forced=None) -> dict | None:
    pts = sorted((p for p in self.points
                  if (p.library, p.op, p.ranks, p.nodes) == (library, op, ranks, nodes)
                  and p.forced == forced and (gpu is None or p.gpu == gpu)),
                 key=lambda p: p.bytes)
    pts = [p for i, p in enumerate(pts) if i == 0 or p.bytes != pts[i - 1].bytes]
    if not pts:
      return None
    if nbytes <= pts[0].bytes:
      p, time = pts[0], pts[0].time_us
    elif nbytes >= pts[-1].bytes:
      p = pts[-1]
      time = p.time_us * nbytes / p.bytes  # bandwidth-bound beyond the sweep
    else:
      i = next(i for i, q in enumerate(pts) if q.bytes >= nbytes)
      a, b = pts[i - 1], pts[i]
      f = math.log(nbytes / a.bytes) / math.log(b.bytes / a.bytes)
      time = math.exp(math.log(a.time_us) + f * math.log(b.time_us / a.time_us))
      p = a if f < 0.5 else b
    return {"time_us": time, "busbw_gbs": bus_factor(op, ranks) * nbytes / time / 1e3,
            "algorithm": (p.algorithm or "-").lower() + (f"/{p.protocol.lower()}"
                                                          if p.protocol else ""),
            "channels": p.channels, "bound": "measured", "source": p.source}


@functools.cache
def measured() -> Table:
  """The tables measured with JAX-Titan that ship with it (planning/data)."""
  points = []
  for path in sorted(DATA.glob("*.json")):
    points += Table.load(path).points
  return Table(points)


def predict(op: str, nbytes: float, ranks: int, nodes: int = 1, *, gpu: gpus_lib.GPU,
            library: str = "rccl", table: Table | None = None, **kwargs) -> dict:
  """A measured point where `table` (default: the shipped measurements) covers the
  group, GPU and size, else the model."""
  table = measured() if table is None else table
  if ranks > 1:
    found = table.lookup(library, "all_to_all" if op in ("dispatch", "combine")
                         and library == "rccl" else op, ranks, nodes, nbytes, gpu.name)
    if found:
      return found
  return model(op, nbytes, ranks, nodes, gpu=gpu, library=library, **kwargs)


_SIZE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(-?\d+)\s+(.*)$")


def parse_rccl_tests(text: str, source: str = "", forced: str | None = None) -> list[Point]:
  """Points from rccl-tests' standard output (with -A 1 for algorithm, protocol and
  channels)."""
  binary = re.search(r"Collective test starting: (\w+)_perf", text)
  ngpus = re.search(r"nGpus\s+(\d+)", text)
  if not binary or not ngpus:
    return []
  op = _RCCL_NAMES.get(binary.group(1), binary.group(1))
  hosts = set(re.findall(r"Rank\s+\d+ Group\s+\d+ Pid\s+\d+ on (\S+) device", text))
  ranks = len(re.findall(r"#\s+Rank\s+\d+ Group", text)) or int(ngpus.group(1))
  gpu = re.search(r"device\s+\d+ \[[0-9a-f:.]+\] (.+)", text)
  out = []
  for line in text.splitlines():
    m = _SIZE.match(line)
    if not m:
      continue
    rest = m.group(6).split()
    try:
      time, busbw = float(rest[0]), float(rest[2])
    except (ValueError, IndexError):
      continue
    algo = proto = channels = None
    if len(rest) >= 11 and rest[-3] != "N/A" and not re.match(r"^[\d.]+$", rest[-3]):
      algo, proto = rest[-3], rest[-2]
      channels = int(rest[-1]) if rest[-1].isdigit() else None
    out.append(Point("rccl", op, ranks, max(1, len(hosts)), int(m.group(1)), time, busbw,
                     algo, proto, channels, forced, gpu.group(1).strip() if gpu else None,
                     source))
  return out


def parse_inspector(rows: list[dict], gpu: str | None = None, source: str = "") -> list[Point]:
  """Points from `jaxtitan.observability.collectives.summarize` of a training run."""
  out = []
  for r in rows:
    op = _RCCL_NAMES.get(str(r["op"]).lower(), str(r["op"]).lower())
    if r.get("bytes") and r.get("time_us_p50"):
      out.append(Point("rccl", op, r["ranks"], r.get("nodes") or 1, r["bytes"],
                       r["time_us_p50"], r.get("busbw_gbs_p50"), gpu=gpu,
                       source=source or "RCCL Inspector"))
  return out


_DTYPE_BYTES = {"float8_e4m3fnuz": 1, "float8_e4m3fn": 1, "fp8": 1, "bfloat16": 2,
                "float16": 2, "float32": 4}


def parse_mori_perf(lines, gpu: str | None = None, source: str = "MORI perf") -> list[Point]:
  """Points from MORI benchmarks' MORI_PERF_OUT records (intra_ep and internode_ep)."""
  out = []
  for line in lines:
    if not line.strip():
      continue
    r = json.loads(line)
    if r.get("category") not in ("intra_ep", "internode_ep"):
      continue
    p, m = r["params"], r["metrics"]
    ranks = p["world_size"]
    nodes = p.get("num_nodes") or (1 if r["category"] == "intra_ep" else max(1, ranks // 8))
    per_token = p["hidden_dim"] * p["num_experts_per_token"]
    for phase, dtype in (("dispatch", p.get("dtype")), ("combine", p.get("combine_dtype"))):
      nbytes = p["max_tokens"] * per_token * _DTYPE_BYTES.get(str(dtype), 2)
      if m.get(f"{phase}_lat_us"):
        out.append(Point("mori", phase, ranks, nodes, nbytes, m[f"{phase}_lat_us"],
                         m.get(f"{phase}_bw_gbps"), gpu=gpu or r.get("platform"),
                         source=source))
  return out


def calibrate(directory) -> Table:
  """Every rccl-tests output, MORI perf record and Inspector summary under `directory`."""
  points = []
  for path in sorted(pathlib.Path(directory).rglob("*")):
    if path.suffix == ".out":
      forced = re.search(r"_(ring|tree)\.out$", path.name)
      points += parse_rccl_tests(path.read_text(errors="replace"), path.name,
                                 forced.group(1).upper() if forced else None)
    elif path.name.endswith(".jsonl") and "mori" in path.name:
      points += parse_mori_perf(path.read_text().splitlines(), source=path.name)
  return Table(points)


def groups(points: list[Point]) -> list[dict]:
  """Per library, operation and group: the latency floor and the peak bus bandwidth."""
  out = {}
  for p in points:
    out.setdefault((p.library, p.op, p.ranks, p.nodes, p.forced, p.gpu), []).append(p)
  rows = []
  for (library, op, ranks, nodes, forced, gpu), pts in sorted(out.items(), key=str):
    best = max(pts, key=lambda p: p.busbw_gbs or 0.0)
    algos = sorted({f"{p.algorithm}/{p.protocol}" for p in pts if p.algorithm})
    rows.append({"library": library, "op": op, "ranks": ranks, "nodes": nodes,
                 "forced": forced, "gpu": gpu, "latency_us": min(p.time_us for p in pts),
                 "peak_busbw_gbs": best.busbw_gbs, "at_bytes": best.bytes,
                 "algorithms": algos, "points": len(pts)})
  return rows


def _size(text: str) -> float:
  m = re.fullmatch(r"([\d.]+)\s*([KMGT]?)i?B?", text.strip(), re.I)
  if not m:
    raise ValueError(f"bad size {text!r}")
  return float(m.group(1)) * 1024 ** ("KMGT".find(m.group(2).upper()) + 1 if m.group(2) else 0)


def _bytes_label(n: float) -> str:
  for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
    if n >= scale:
      return f"{n / scale:g} {unit}"
  return f"{n:g} B"


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--op", default="all_reduce", choices=OPS)
  parser.add_argument("--ranks", type=int, default=8)
  parser.add_argument("--nodes", type=int, default=1)
  parser.add_argument("--gpu", default="MI355X")
  parser.add_argument("--library", default=None, choices=["rccl", "mori"],
                      help="Default: MORI for dispatch and combine, RCCL otherwise.")
  parser.add_argument("--min_bytes", default="1K")
  parser.add_argument("--max_bytes", default="4G")
  parser.add_argument("--tokens", type=int, default=0,
                      help="dispatch and combine: tokens per rank, instead of sizes.")
  parser.add_argument("--hidden", type=int, default=7168)
  parser.add_argument("--topk", type=int, default=8)
  parser.add_argument("--table", default="",
                      help="Measured points from --calibrate; default: the shipped ones.")
  parser.add_argument("--model_only", action="store_true",
                      help="Ignore measured points: the link model alone.")
  parser.add_argument("--scaleout_gbs", type=float, default=None)
  parser.add_argument("--efficiency", type=float, default=EFFICIENCY)
  parser.add_argument("--calibrate", default="",
                      help="A scripts/bench_collectives.sh output directory to parse.")
  parser.add_argument("--json", default="")
  parser.add_argument("--compact", action="store_true",
                      help="--calibrate: write points only, unindented (planning/data).")
  args = parser.parse_args(argv)

  if args.calibrate:
    table = calibrate(args.calibrate)
    if args.json:
      table.save(args.json, compact=args.compact)
    print(f"{'library':8s} {'op':15s} {'ranks':>5s} {'nodes':>5s} {'forced':>6s} "
          f"{'latency us':>10s} {'peak busbw':>10s} {'at':>9s}  algorithms")
    for g in groups(table.points):
      print(f"{g['library']:8s} {g['op']:15s} {g['ranks']:5d} {g['nodes']:5d} "
            f"{g['forced'] or '-':>6s} {g['latency_us']:10.1f} {g['peak_busbw_gbs'] or 0:10.1f} "
            f"{_bytes_label(g['at_bytes']):>9s}  {', '.join(g['algorithms'])}")
    return 0

  gpu = gpus_lib.get(args.gpu)
  library = args.library or ("mori" if args.op in ("dispatch", "combine") else "rccl")
  table = Table([]) if args.model_only else (Table.load(args.table) if args.table else None)
  if args.tokens:
    per = args.hidden * args.topk * (1 if args.op == "dispatch" else 2)
    sizes = [args.tokens * per]
  else:
    lo, hi = _size(args.min_bytes), _size(args.max_bytes)
    sizes = [lo * 2**i for i in range(int(math.log2(hi / lo)) + 1)]
  rows = [{"bytes": s, **predict(args.op, s, args.ranks, args.nodes, gpu=gpu, library=library,
                                 table=table, scaleout_gbs=args.scaleout_gbs,
                                 efficiency=args.efficiency)} for s in sizes]
  print(f"{library} {args.op}, {args.ranks} ranks on {args.nodes} node(s) of {gpu.name}")
  print(f"{'size':>10s} {'algorithm':>12s} {'time us':>10s} {'busbw GB/s':>10s}  source")
  for r in rows:
    print(f"{_bytes_label(r['bytes']):>10s} {r['algorithm']:>12s} {r['time_us']:10.1f} "
          f"{r['busbw_gbs']:10.1f}  {r['source']}")
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
