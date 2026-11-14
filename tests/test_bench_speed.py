"""dub bench speed: union of overlapping spans, GPU samples lined up by wall clock."""

from __future__ import annotations

import json

from dub.bench.speed import _union, analyse


def test_union_counts_overlap_once():
    assert _union([(0, 2), (1, 3), (5, 6)]) == 4
    assert _union([]) == 0


def test_stages_and_gpu_samples_line_up(tmp_path):
    events = [
        {"ph": "M", "pid": 1, "tid": 100, "name": "thread_name", "args": {"name": "stages"}},
        {"ph": "M", "pid": 1, "tid": 200, "name": "thread_name", "args": {"name": "gpu · tts"}},
        {"ph": "i", "pid": 1, "tid": 100, "name": "[1/7] a", "ts": 0},
        {"ph": "i", "pid": 1, "tid": 100, "name": "[2/7] b", "ts": 2e6},
        {"ph": "X", "pid": 1, "tid": 200, "name": "tts.synthesize", "ts": 2e6, "dur": 2e6},
    ]
    (tmp_path / "trace.json").write_text(
        json.dumps({"traceEvents": events, "otherData": {"t0_unix": 1000.0}}), encoding="utf-8"
    )
    (tmp_path / "gpu.csv").write_text(
        "unix_s,util_pct,mem_mib\n1000.5,0,1\n1001.5,0,1\n1002.5,100,1\n1003.5,80,1\n",
        encoding="utf-8",
    )
    rep = analyse(tmp_path, media_s=8.0)
    assert rep.wall_s == 4.0 and rep.gpu_util_mean == 45.0
    assert [s["gpu_util"] for s in rep.stages] == [0.0, 90.0]
    assert rep.lanes[0] == {"lane": "gpu · tts", "busy_s": 2.0, "busy_share": 0.5}
    assert rep.as_dict()["speed_x"] == 2.0
