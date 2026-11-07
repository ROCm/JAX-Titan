"""Collective completion times from the RCCL Inspector.

  python -m jaxtitan.observability.collectives DIR [--json OUT]

The Inspector, an RCCL profiler plugin (`scripts/build_rccl_inspector.sh`),
writes one JSON object per completed collective or point-to-point operation
for each rank into DIR when the trainer runs with `--rccl_inspector`. This
groups them by operation, communicator size and message size, and reports
the count and completion time p50 and p99 with the p50 bus bandwidth. The
slowest flow sets a collective's completion time, so the tail says more than
mean bandwidth does (the ROCm scaling book, Chapter 6).
"""

import argparse
import json
import math
import pathlib


def _quantile(values, q):
  values = sorted(values)
  pos = (len(values) - 1) * q
  lo, hi = math.floor(pos), math.ceil(pos)
  return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _objects(path: pathlib.Path):
  """JSON objects from a file holding one per line, or several concatenated."""
  text = path.read_text()
  decoder = json.JSONDecoder()
  i = 0
  while i < len(text):
    while i < len(text) and text[i].isspace():
      i += 1
    if i >= len(text):
      break
    obj, i = decoder.raw_decode(text, i)
    yield obj


def operations(directory) -> list[dict]:
  out = []
  for path in sorted(pathlib.Path(directory).rglob("*")):
    if not path.is_file() or path.suffix not in (".log", ".json", ".jsonl", ""):
      continue
    try:
      objects = list(_objects(path))
    except (json.JSONDecodeError, UnicodeDecodeError):
      continue
    for obj in objects:
      header = obj.get("header", {})
      for kind, name_key in (("coll_perf", "coll"), ("p2p_perf", "p2p")):
        perf = obj.get(kind)
        if not perf:
          continue
        out.append({
            "op": perf.get(name_key),
            "timing": perf.get(f"{name_key}_timing_source"),
            "ranks": header.get("n_ranks"),
            "nodes": header.get("nnodes"),
            "rank": header.get("rank"),
            "comm": header.get("id"),
            "bytes": perf.get("coll_msg_size_bytes") or perf.get("p2p_msg_size_bytes"),
            "time_us": perf.get("coll_exec_time_us") or perf.get("p2p_exec_time_us"),
            "busbw_gbs": perf.get("coll_busbw_gbs") or perf.get("p2p_busbw_gbs"),
        })
  return [o for o in out if o["time_us"] is not None]


def summarize(directory) -> list[dict]:
  groups = {}
  for o in operations(directory):
    groups.setdefault((o["op"], o["ranks"], o["nodes"], o["bytes"]), []).append(o)
  out = []
  for (op, ranks, nodes, size), ops in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][3] or 0)):
    times = [o["time_us"] for o in ops]
    bws = [o["busbw_gbs"] for o in ops if o["busbw_gbs"] is not None]
    out.append({
        "op": op, "ranks": ranks, "nodes": nodes, "bytes": size, "count": len(ops),
        "timing": sorted({str(o["timing"]) for o in ops}),
        "time_us_p50": _quantile(times, 0.5), "time_us_p99": _quantile(times, 0.99),
        "time_us_max": max(times), "busbw_gbs_p50": _quantile(bws, 0.5) if bws else None,
    })
  return out


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directory")
  parser.add_argument("--json", default="")
  args = parser.parse_args(argv)
  rows = summarize(args.directory)
  print("op            ranks  bytes          count   p50 us    p99 us   busbw p50 GB/s")
  for r in rows:
    print(f"{str(r['op']):12s}  {r['ranks'] or '-':>5}  {r['bytes'] or 0:>13}  {r['count']:>5}  "
          f"{r['time_us_p50']:8.1f}  {r['time_us_p99']:8.1f}  "
          f"{r['busbw_gbs_p50'] if r['busbw_gbs_p50'] is not None else float('nan'):8.1f}")
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
