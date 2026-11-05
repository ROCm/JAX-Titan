"""Per-GPU busy and idle time from a `jax.profiler` trace.

  python -m jaxtitan.observability.profile TRACE_DIR [--lines]

Reads the newest `*.xplane.pb` under TRACE_DIR and reports, for each GPU, the
share of the traced window it spends in compute kernels, in collective
kernels and in memory copies, and idle. The window runs from the first to the
last GPU event, so a pipeline stage's idle share is its bubble plus any time
it waits for transfers. Overlapping events count once, compute first.
"""

import argparse
import pathlib
import re

COLLECTIVE = re.compile(r"nccl|rccl|all.?reduce|all.?gather|reduce.?scatter|all.?to.?all|collective.?permute", re.I)
# Runtime copies only: XLA's own copy fusions are compute kernels.
COPY = re.compile(r"^Memcpy|^Memset|__amd_rocclr_(copy|fill)", re.I)
DEVICE_PLANE = re.compile(r"/device:GPU:(\d+)")


def _merge(intervals):
  out = []
  for start, end in sorted(intervals):
    if out and start <= out[-1][1]:
      out[-1][1] = max(out[-1][1], end)
    else:
      out.append([start, end])
  return out


def _length(intervals):
  return sum(end - start for start, end in intervals)


def _subtract(intervals, covered):
  """Parts of merged `intervals` not in merged `covered`."""
  out, j = [], 0
  for start, end in intervals:
    while j < len(covered) and covered[j][1] <= start:
      j += 1
    k, cur = j, start
    while k < len(covered) and covered[k][0] < end:
      if covered[k][0] > cur:
        out.append([cur, covered[k][0]])
      cur = max(cur, covered[k][1])
      k += 1
    if cur < end:
      out.append([cur, end])
  return out


def _is_stream_line(name: str) -> bool:
  return name.startswith("Stream") or "stream" in name.lower()


def device_events(xplane: pathlib.Path):
  """{gpu id: [(name, start_ns, end_ns)]} from the GPU stream lines."""
  from jax.profiler import ProfileData  # pylint: disable=g-import-not-at-top

  data = ProfileData.from_file(str(xplane))
  events = {}
  for plane in data.planes:
    m = DEVICE_PLANE.search(plane.name)
    if not m:
      continue
    out = events.setdefault(int(m.group(1)), [])
    for line in plane.lines:
      if _is_stream_line(line.name):
        out.extend((e.name, e.start_ns, e.end_ns) for e in line.events)
  return events


def summarize(xplane: pathlib.Path) -> dict:
  """{gpu id: {"compute", "collective", "copy", "idle"} as window fractions}."""
  events = device_events(xplane)
  starts = [s for evs in events.values() for _, s, _ in evs]
  ends = [e for evs in events.values() for _, _, e in evs]
  if not starts:
    raise ValueError(f"no GPU stream events in {xplane}")
  window = max(ends) - min(starts)
  result = {}
  for gpu, evs in sorted(events.items()):
    kinds = {"compute": [], "collective": [], "copy": []}
    for name, start, end in evs:
      kind = "collective" if COLLECTIVE.search(name) else (
          "copy" if COPY.search(name) else "compute")
      kinds[kind].append((start, end))
    compute = _merge(kinds["compute"])
    collective = _subtract(_merge(kinds["collective"]), compute)
    copy = _subtract(_subtract(_merge(kinds["copy"]), compute), _merge(collective))
    busy = _length(compute) + _length(collective) + _length(copy)
    result[gpu] = {
        "compute": _length(compute) / window,
        "collective": _length(collective) / window,
        "copy": _length(copy) / window,
        "idle": 1 - busy / window,
        "window_s": window / 1e9,
    }
  return result


def newest_xplane(trace_dir: str) -> pathlib.Path:
  files = sorted(pathlib.Path(trace_dir).rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)
  if not files:
    raise FileNotFoundError(f"no *.xplane.pb under {trace_dir}")
  return files[-1]


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("trace_dir")
  parser.add_argument(
      "--lines", action="store_true",
      help="List planes, lines and their most frequent event names instead.",
  )
  args = parser.parse_args(argv)
  xplane = newest_xplane(args.trace_dir)
  print(xplane)
  if args.lines:
    import collections  # pylint: disable=g-import-not-at-top
    from jax.profiler import ProfileData  # pylint: disable=g-import-not-at-top

    for plane in ProfileData.from_file(str(xplane)).planes:
      lines = list(plane.lines)
      print(f"plane {plane.name}: {len(lines)} lines")
      for line in lines:
        names = collections.Counter(e.name for e in line.events)
        top = ", ".join(f"{n[:60]} x{c}" for n, c in names.most_common(4))
        print(f"  line {line.name!r}: {sum(names.values())} events; {top}")
    return 0
  result = summarize(xplane)
  print(f"window {next(iter(result.values()))['window_s']:.3f} s")
  print("gpu  compute  collective  copy   idle")
  for gpu, r in result.items():
    print(
        f"{gpu:3d}  {100 * r['compute']:6.1f}%  {100 * r['collective']:9.1f}%  "
        f"{100 * r['copy']:4.1f}%  {100 * r['idle']:5.1f}%"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
