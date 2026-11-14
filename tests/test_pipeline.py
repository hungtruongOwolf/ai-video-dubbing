"""End to end on the laptop: demo video -> fake dub -> valid video with 2 audio + 2 subtitle
tracks, and a rerun that is served 100% from the cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dub.cli import main
from dub.ops.media import probe
from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    from dub.demo import make_demo

    video, _ = make_demo(tmp_path_factory.mktemp("demo"))
    return video


def run(video: Path, ws: Path, *extra: str) -> int:
    return main(["run", str(video), "--to", "vi", "--fake", "--workspace", str(ws), *extra])


def last_report(ws: Path) -> str:
    reports = sorted((ws / "jobs").glob("*/report.md"), key=lambda p: p.stat().st_mtime)
    return reports[-1].read_text(encoding="utf-8")


def test_fake_dub_produces_a_valid_video_and_reruns_from_cache(demo, tmp_path, capsys):
    ws = tmp_path / "ws"
    assert run(demo, ws) == 0
    first = capsys.readouterr().out
    out = demo.parent / "interview.dub" / "interview.vi.mp4"
    assert out.exists()

    streams = probe(out)["streams"]
    kinds = [s["codec_type"] for s in streams]
    assert kinds.count("video") == 1 and kinds.count("audio") == 2 and kinds.count("subtitle") == 2
    langs = [s.get("tags", {}).get("language") for s in streams if s["codec_type"] == "audio"]
    assert langs == ["vie", "eng"]
    src_dur = float(probe(demo)["format"]["duration"])
    assert abs(float(probe(out)["format"]["duration"]) - src_dur) < 0.5

    report = last_report(ws)
    assert "| tts.synthesize | fake_tts |" in report
    assert "## Cast sheet" in report

    assert run(demo, ws) == 0
    second = capsys.readouterr().out
    calls = [ln for ln in second.splitlines() if ln.startswith("calls:")][0]
    n, hits = [int(x) for x in calls.replace(",", "").split() if x.isdigit()]
    assert n == hits, calls
    assert "calls:" in first


def test_job_folder_has_review_files(demo, tmp_path):
    ws = tmp_path / "ws"
    assert run(demo, ws) == 0
    job = next((ws / "jobs").iterdir())
    for name in ("report.md", "trace.json", "routes.jsonl", "cast.json", "units.json"):
        assert (job / name).exists(), name
    trace = json.loads((job / "trace.json").read_text(encoding="utf-8"))
    names = {e["name"] for e in trace["traceEvents"]}
    assert "tts.synthesize" in names and "text.segment" in names
    units = json.loads((job / "units.json").read_text(encoding="utf-8"))
    assert all(u["placement"]["audio_key"] for u in units)
    assert all(u["unit"]["end_ms"] - u["unit"]["start_ms"] <= 15000 for u in units)
    # M3: every line is heard again (the fake ASR hears exactly the text) and measured.
    assert all(u["backcheck_cer"] == 0.0 for u in units)
    m = json.loads((job / "metrics.json").read_text(encoding="utf-8"))
    assert m["units_over_15s"] == 0 and m["onset_err_ms_p95"] >= 0 and m["drift_ms"] == 0
    assert m["speedup_p95"] >= 1.0 and m["backcheck_cer_p50"] == 0.0
    assert "## Sync and quality" in (job / "report.md").read_text(encoding="utf-8")
    seg = json.loads((job / "segmentation.json").read_text(encoding="utf-8"))
    assert seg["units"] and seg["sentences"]


def test_exact_duration_engine_and_subs_only(demo, tmp_path):
    ws = tmp_path / "ws"
    assert run(demo, ws, "--use", "tts.synthesize=fake_tts_exact") == 0
    assert "| tts.synthesize | fake_tts_exact |" in last_report(ws)
    assert run(demo, ws, "--subs-only", "--out-dir", str(tmp_path / "subs")) == 0
    assert (tmp_path / "subs" / "interview.vi.srt").exists()
    assert not (tmp_path / "subs" / "interview.vi.mp4").exists()


def test_without_fake_the_error_explains_what_is_missing(demo, tmp_path, capsys):
    assert main(["run", str(demo), "--to", "vi", "--workspace", str(tmp_path / "ws")]) == 3
    err = capsys.readouterr().err
    assert "No runnable provider" in err and "--fake" in err


def test_bench_seg_analyses_a_video_without_dubbing(demo, tmp_path, capsys):
    import re

    from dub.demo import SCRIPT

    sentences = []
    for _spk, text, _pause in SCRIPT:
        sentences += [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s]
    golden = tmp_path / "golden.txt"
    golden.write_text("\n".join(sentences), encoding="utf-8")
    scores = tmp_path / "scores.json"
    ws = tmp_path / "ws"
    args = ["bench", "seg", str(demo), "--golden", str(golden), "--from", "en", "--fake"]
    assert main([*args, "--workspace", str(ws), "--json", str(scores)]) == 0
    out = capsys.readouterr().out
    assert "translating" not in out and "voicing" not in out  # phase A only
    s = json.loads(scores.read_text(encoding="utf-8"))
    assert s["tol1"]["n_gold"] > 5 and s["tol1"]["f1"] > 0.8  # rules on a punctuated replay
    assert (Path(s["job"]) / "segmentation.json").exists() and "analysis" in s["job"]


def test_final_profile_runs_the_qa_gate_take_by_take(demo, tmp_path):
    ws = tmp_path / "ws"
    assert run(demo, ws, "--profile", "final", "--speakers", "2") == 0
    job = next(p for p in (ws / "jobs").iterdir() if "final" in p.name)
    units = json.loads((job / "units.json").read_text(encoding="utf-8"))
    assert all(u["qa"] and u["qa"]["passed"] for u in units)
    m = json.loads((job / "metrics.json").read_text(encoding="utf-8"))
    assert m["qa_passed"] == 1.0 and m["speakers"] >= 1 and "ref_sim" in m


def test_confirm_cast_stops_before_translation_then_corrections_apply(demo, tmp_path, capsys):
    ws = tmp_path / "ws"
    assert run(demo, ws, "--confirm-cast") == 0
    out = capsys.readouterr().out
    assert "# Cast sheet" in out and "translating" not in out and "dub edit" in out
    job = next((ws / "jobs").iterdir())
    assert (job / "cast.md").exists() and not (job / "units.json").exists()
    assert main(["edit", str(demo), "--to", "vi", "--workspace", str(ws),
                 "cast", "S1", "name=Ana", "gender=female"]) == 0  # fmt: skip
    capsys.readouterr()
    assert run(demo, ws) == 0
    out = capsys.readouterr().out
    assert "translating" in out
    cast = json.loads((job / "cast.json").read_text(encoding="utf-8"))["cast"]
    s1 = next(s for s in cast["speakers"] if s["id"] == "S1")
    assert s1["name"] == "Ana" and s1["gender"] == "female" and s1["locked"]


def test_bench_speed_reports_stages_lanes_and_ops(demo, tmp_path, capsys):
    out_json = tmp_path / "speed.json"
    args = ["bench", "speed", str(demo), "--fake", "--workspace", str(tmp_path / "ws")]
    assert main([*args, "--json", str(out_json)]) == 0
    text = capsys.readouterr().out
    assert "[6/7] voicing" in text and "tts.synthesize" in text
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert [s["stage"][:5] for s in rep["stages"]] == [f"[{k}/7]" for k in range(1, 8)]
    assert rep["wall_s"] > 0 and rep["media_s"] > 50 and rep["gpu_util_mean"] is None
    assert sum(s["wall_s"] for s in rep["stages"]) <= rep["wall_s"] + 0.5
    # --fresh: a second run recomputes everything instead of reading the cache
    assert main([*args, "--fresh"]) == 0
    job = next((tmp_path / "ws" / "jobs").iterdir())
    names = [e["name"] for e in json.loads((job / "trace.json").read_text())["traceEvents"]]
    assert "tts.synthesize" in names and not any(n.endswith("(cache)") for n in names)


def test_one_source_many_targets_shares_the_analysis(demo, tmp_path, capsys):
    ws = tmp_path / "ws"
    out_dir = tmp_path / "multi"
    args = ["run", str(demo), "--to", "vi,es", "--fake", "--workspace", str(ws)]
    assert main([*args, "--out-dir", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "[vi] " in out and "[es] " in out
    assert (out_dir / "interview.vi.mp4").exists() and (out_dir / "interview.es.mp4").exists()
    jobs = sorted(p.name for p in (ws / "jobs").iterdir())
    assert any("-vi-" in j for j in jobs) and any("-es-" in j for j in jobs)
