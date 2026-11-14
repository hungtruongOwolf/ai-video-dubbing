"""Where the time of a job goes (`dub bench speed`): stages, ops, lanes and GPU use.

Inputs, all in the job folder: `trace.json` (op spans, stage marks, wall clock of ts=0) and,
for pod jobs, `gpu.csv` (utilisation samples from dubd). Outputs:
- per stage (between the "[k/7]" marks): wall time and mean GPU utilisation;
- per op: calls, busy time (sum of spans) and the wall-clock window it spans;
- per lane ("gpu · tts", "llm", ...): busy time as a union of spans (overlap counted once);
- the whole job: wall time, GPU utilisation, and the media minutes dubbed per wall minute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    name: str
    lane: str
    start_s: float
    end_s: float


@dataclass
class SpeedReport:
    wall_s: float
    stages: list[dict[str, Any]] = field(default_factory=list)
    ops: list[dict[str, Any]] = field(default_factory=list)
    lanes: list[dict[str, Any]] = field(default_factory=list)
    gpu_util_mean: float | None = None
    media_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_s": round(self.wall_s, 1),
            "media_s": self.media_s,
            "speed_x": round(self.media_s / self.wall_s, 2) if self.media_s else None,
            "gpu_util_mean": self.gpu_util_mean,
            "stages": self.stages,
            "lanes": self.lanes,
            "ops": self.ops,
        }


def _union(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def load_trace(path: Path) -> tuple[list[Span], list[tuple[float, str]], float | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lanes = {
        e["tid"]: e["args"]["name"].split(" #")[0]
        for e in data["traceEvents"]
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    spans, marks = [], []
    for e in data["traceEvents"]:
        if e.get("ph") == "X":
            s = e["ts"] / 1e6
            spans.append(Span(e["name"], lanes.get(e["tid"], "?"), s, s + e["dur"] / 1e6))
        elif e.get("ph") == "i" and lanes.get(e["tid"]) == "stages":
            marks.append((e["ts"] / 1e6, e["name"]))
    t0 = (data.get("otherData") or {}).get("t0_unix")
    return spans, sorted(marks), t0


def load_gpu(path: Path, t0_unix: float | None) -> list[tuple[float, float]]:
    """(seconds since the trace's ts=0, utilisation %) samples."""
    if t0_unix is None or not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        t, util, _mem = line.split(",")
        out.append((float(t) - t0_unix, float(util)))
    return out


def _gpu_mean(samples: list[tuple[float, float]], s: float, e: float) -> float | None:
    sel = [u for t, u in samples if s <= t <= e]
    return round(sum(sel) / len(sel), 1) if sel else None


def analyse(job_dir: Path, media_s: float | None = None) -> SpeedReport:
    spans, marks, t0 = load_trace(job_dir / "trace.json")
    gpu = load_gpu(job_dir / "gpu.csv", t0)
    end = max([sp.end_s for sp in spans] + [m[0] for m in marks] + [0.0])
    rep = SpeedReport(wall_s=end, media_s=media_s)
    rep.gpu_util_mean = _gpu_mean(gpu, 0.0, end)

    bounds = [*marks, (end, "end")]
    for (s, name), (e, _) in zip(bounds, bounds[1:], strict=False):
        rep.stages.append({
            "stage": name, "start_s": round(s, 1), "wall_s": round(e - s, 1),
            "gpu_util": _gpu_mean(gpu, s, e),
        })  # fmt: skip

    by_op: dict[str, list[Span]] = {}
    for sp in spans:
        by_op.setdefault(sp.name, []).append(sp)
    for name, sps in sorted(by_op.items(), key=lambda kv: -sum(x.end_s - x.start_s for x in kv[1])):
        rep.ops.append({
            "op": name, "lane": sps[0].lane, "calls": len(sps),
            "busy_s": round(sum(x.end_s - x.start_s for x in sps), 1),
            "window_s": [round(min(x.start_s for x in sps), 1),
                         round(max(x.end_s for x in sps), 1)],
        })  # fmt: skip

    by_lane: dict[str, list[tuple[float, float]]] = {}
    for sp in spans:
        by_lane.setdefault(sp.lane, []).append((sp.start_s, sp.end_s))
    for lane, iv in sorted(by_lane.items(), key=lambda kv: -_union(kv[1])):
        busy = _union(iv)
        rep.lanes.append({
            "lane": lane, "busy_s": round(busy, 1),
            "busy_share": round(busy / end, 3) if end else 0.0,
        })  # fmt: skip
    return rep


def render(rep: SpeedReport) -> str:
    d = rep.as_dict()
    head = f"wall {d['wall_s']} s"
    if d["media_s"]:
        head += f" for {d['media_s'] / 60:.1f} min of media ({d['speed_x']}x real time)"
    if d["gpu_util_mean"] is not None:
        head += f", GPU utilisation {d['gpu_util_mean']}%"
    out = [head, "", f"{'stage':<50} {'wall s':>7} {'GPU %':>7}"]
    for st in d["stages"]:
        g = "" if st["gpu_util"] is None else f"{st['gpu_util']:7.1f}"
        out.append(f"{st['stage'][:50]:<50} {st['wall_s']:7.1f} {g}")
    out += ["", "lane                      busy s   share of wall"]
    for ln in d["lanes"]:
        out.append(f"{ln['lane']:<24} {ln['busy_s']:8.1f}   {ln['busy_share']:.0%}")
    out += ["", "op                        calls    busy s   window s"]
    for op in d["ops"][:15]:
        w = op["window_s"]
        out.append(f"{op['op']:<24} {op['calls']:6d} {op['busy_s']:9.1f}   {w[0]}-{w[1]}")
    return "\n".join(out)
