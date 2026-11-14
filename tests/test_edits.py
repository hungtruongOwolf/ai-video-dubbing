"""M2: human edits, partial re-render, tiles, risk, the review app, kill-and-resume."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from dub.contracts.v1 import Edit, Placement, Translation, Unit, Word
from dub.edits import EditStore
from dub.ops.edits import UnitEdits, apply_structure, effective
from dub.ops.mix import plan_tiles, soft_clip
from dub.ops.risk import score
from dub.runner import RunOptions, run_video
from dub.store import Workspace
from tests.conftest import requires_ffmpeg


def unit(uid, start, end, text, spk="S1"):
    return Unit(
        id=uid,
        speaker=spk,
        sentence_ids=["s"],
        start_ms=start,
        end_ms=end,
        text=text,
        n_words=len(text.split()),
    )


def words(text, start, step=400, spk="S1"):
    return [
        Word(w=w, start_ms=start + i * step, end_ms=start + i * step + 300, speaker=spk)
        for i, w in enumerate(text.split())
    ]


def ed(i, uid, field, value=None):
    return Edit(id=f"{i:020d}-0000", unit_id=uid, field=field, value=value)


# -------------------------------------------------------------------- pure edit logic


def test_undo_cancels_an_edit():
    es = [ed(1, "u1", "text", "a"), ed(2, "u1", "text", "b"), ed(3, "u1", "undo", f"{2:020d}-0000")]
    assert [e.value for e in effective(es)] == ["a"]


def test_split_and_merge_give_deterministic_ids():
    us = [unit("u1", 0, 1900, "one two three four five"), unit("u2", 2500, 3300, "six seven")]
    ws = words("one two three four five", 0) + words("six seven", 2500)
    out, warn = apply_structure(us, ws, [ed(1, "u1", "split", {"at_ms": 1000})])
    assert [u.id for u in out] == ["u1.a", "u1.b", "u2"] and not warn
    assert out[0].text == "one two three" and out[1].text == "four five"
    assert out[0].parent_ids == ["u1"]
    merged, _ = apply_structure(out, ws, [ed(2, "u1.b", "merge_next")])
    assert [u.id for u in merged] == ["u1.a", "u1.b+u2"]
    assert merged[1].text == "four five six seven" and merged[1].parent_ids == ["u1", "u2"]


def test_structure_edits_that_no_longer_apply_are_reported():
    us = [unit("u1", 0, 1000, "a b", "S1"), unit("u2", 1200, 2000, "c d", "S2")]
    out, warn = apply_structure(us, [], [ed(1, "gone", "split"), ed(2, "u1", "merge_next")])
    assert out == us and len(warn) == 2


def test_unit_edits_speaker_nudge_text_lock():
    us = [unit("u1", 0, 1000, "a"), unit("u2", 1500, 2500, "b"), unit("u3", 3000, 4000, "c")]
    ue = UnitEdits.from_edits(
        [
            ed(1, "u2", "speaker", "S2"),
            ed(2, "u2", "nudge", {"start_ms": -800, "end_ms": 300}),
            ed(3, "u3", "text", "hello"),
            ed(4, "u1", "lock", True),
        ]
    )
    out = ue.apply_to_units(us)
    assert out[1].speaker == "S2"
    assert (out[1].start_ms, out[1].end_ms) == (1000, 2800)  # clamped at the previous line's end
    assert ue.is_protected("u3") and ue.is_protected("u1") and not ue.is_protected("u2")
    tr = ue.apply_to_translation(Translation(unit_id="u3", short="x", target="y", long="z"))
    assert (tr.short, tr.target, tr.long) == ("hello", "hello", "hello")


def test_edit_store_appends_and_undoes(tmp_path):
    store = EditStore(tmp_path / "e.jsonl")
    a = store.add("u1", "text", "hi")
    store.add("u1", "seed", 1)
    store.undo()  # cancels the seed
    assert [e.field for e in store.effective()] == ["text"]
    store.undo(a.id)
    assert store.effective() == []
    (tmp_path / "e.jsonl").open("a", encoding="utf-8").write('{"torn line')
    assert len(store.all()) == 4  # a torn last line is ignored


def test_tiles_and_soft_clip():
    assert plan_tiles(65_000) == [(0, 30_000), (30_000, 60_000), (60_000, 65_000)]
    import numpy as np

    x = np.array([0.1, -0.5, 0.95, -3.0], dtype=np.float32)
    y = soft_clip(x)
    assert y[0] == pytest.approx(0.1) and y[1] == pytest.approx(-0.5)
    assert 0.8 < y[2] < 0.95 and -1.0 <= y[3] < -0.8


def test_risk_puts_problems_first():
    u = unit("u1", 0, 3000, "hello there")
    p = Placement(
        unit_id="u1",
        speaker="S1",
        slot_start_ms=0,
        slot_end_ms=3000,
        start_ms=0,
        end_ms=2500,
        natural_ms=3000,
        clip_ms=3000,
        rate=1.2,
        variant="target",
        text="x",
        flags=["overflow", "fast"],
    )
    bad, why = score(u, p, None)
    good, _ = score(u, p.model_copy(update={"rate": 1.0, "flags": []}), None)
    assert bad > good and any("does not fit" in w for w in why)
    assert score(u, p, None, edited=True)[0] < bad


# -------------------------------------------------------------------- end to end (fake models)


class SimulatedKill(BaseException):
    """Like a kill: not an Exception, so nothing in the pipeline catches it."""


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    from dub.demo import make_demo

    video, _ = make_demo(tmp_path_factory.mktemp("demo"))
    return video


@requires_ffmpeg
def test_edit_rerenders_only_the_touched_line(demo, tmp_path):
    ws = Workspace.open(tmp_path / "ws")
    opts = RunOptions(video=demo, tgt="vi", fake=True)
    first = run_video(opts, ws, log=lambda m: None)
    target = json.loads((first.job_dir / "units.json").read_text(encoding="utf-8"))[3]["unit"]
    store = EditStore.for_job(
        ws.root,
        json.loads((ws.root / "sources.json").read_text(encoding="utf-8"))[str(demo.resolve())][
            "audio_key"
        ],
        "vi",
    )

    store.add(target["id"], "text", "Một câu đã sửa tay.")
    second = run_video(opts, ws, log=lambda m: None)
    assert second.redone.get("tts.synthesize") == 1
    assert second.redone.get("audio.mix_tile", 0) <= 2
    assert "mt.translate" not in second.redone and "asr.transcribe" not in second.redone

    store.undo()
    third = run_video(opts, ws, log=lambda m: None)
    assert "tts.synthesize" not in third.redone  # back to the cached original take


@requires_ffmpeg
def test_kill_and_resume_gives_identical_output(demo, tmp_path, monkeypatch):
    """Crash in the middle of TTS, run again: finished steps come from the cache and the result
    is byte-identical to an uninterrupted run."""
    from dub.providers import fake

    ref = run_video(
        RunOptions(video=demo, tgt="vi", fake=True),
        Workspace.open(tmp_path / "ref"),
        log=lambda m: None,
    )

    ws = Workspace.open(tmp_path / "ws")
    real_op = fake.FakeTTS.op_tts_synthesize
    calls = {"n": 0}

    def flaky(self, req, ctx):
        calls["n"] += 1
        if calls["n"] == 7:
            raise SimulatedKill("simulated kill")
        return real_op(self, req, ctx)

    monkeypatch.setattr(fake.FakeTTS, "op_tts_synthesize", flaky)
    with pytest.raises(SimulatedKill):
        run_video(RunOptions(video=demo, tgt="vi", fake=True), ws, log=lambda m: None)
    monkeypatch.setattr(fake.FakeTTS, "op_tts_synthesize", real_op)
    resumed = run_video(RunOptions(video=demo, tgt="vi", fake=True), ws, log=lambda m: None)
    total = ref.redone["tts.synthesize"]  # takes of an uninterrupted run (audition included)
    assert resumed.redone.get("tts.synthesize", 0) < total  # earlier takes were reused
    assert resumed.dub_audio.read_bytes() == ref.dub_audio.read_bytes()


@requires_ffmpeg
def test_review_app_api(demo, tmp_path):
    from http.server import ThreadingHTTPServer

    from dub.review.server import ReviewApp, make_handler

    ws = Workspace.open(tmp_path / "ws")
    app = ReviewApp(RunOptions(video=demo, tgt="vi", fake=True), ws)
    app.start_render(export=False)
    for _ in range(600):
        if not app.render["running"]:
            break
        threading.Event().wait(0.1)
    assert app.render["error"] is None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=60) as r:
            return r.read()

    def post(path, body):
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    assert b"Dub review" in get("/")
    state = json.loads(get("/api/state"))
    assert state["units"] and state["pending"] == 0
    u = state["units"][0]["unit"]
    assert get(f"/api/audio/orig?from={u['start_ms']}&to={u['end_ms']}")[:3] in (
        b"ID3",
        b"\xff\xfb",
    )
    assert len(get(f"/api/audio/dub?from={u['start_ms']}&to={u['end_ms']}")) > 1000
    post("/api/edit", {"unit_id": u["id"], "field": "seed", "value": 1})
    assert json.loads(get("/api/state"))["pending"] == 1
    post("/api/undo", {})
    assert json.loads(get("/api/state"))["edits"] == []
    httpd.shutdown()


def test_cast_corrections_override_and_lock_the_sheet():
    from dub.contracts.v1 import CastSheet, CastSpeaker, ContextSheet, Edit
    from dub.ops.edits import apply_cast

    sheet = ContextSheet(summary="", cast=CastSheet(speakers=[CastSpeaker(id="S2", gender="male")]))
    edits = [
        Edit(id="1", unit_id="cast", field="cast", value={"speaker": "S2", "gender": "female"}),
        Edit(id="2", unit_id="cast", field="cast",
             value={"speaker": "S1", "listener": "S2", "lang": "vi", "self_term": "mẹ",
                    "other_term": "con"}),
    ]  # fmt: skip
    out = apply_cast(sheet, edits).cast
    s2 = next(s for s in out.speakers if s.id == "S2")
    assert s2.gender == "female" and s2.locked
    rel = out.relations[0]
    assert (rel.speaker, rel.listener) == ("S1", "S2") and rel.address["vi"].self_term == "mẹ"
    assert sheet.cast.speakers[0].gender == "male"  # the input is not modified
