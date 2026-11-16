"""Emotion of a line from emotion2vec embeddings (dub.ops.emotion, M13)."""

from __future__ import annotations

from dub.contracts.v1 import RefBank, Unit, VoiceRef
from dub.ops import emotion

LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised",
          "unknown"]  # fmt: skip


def _p(lab: str, p: float = 0.8) -> list[float]:
    rest = (1 - p) / 8
    return [p if x == lab else rest for x in LABELS]


def test_centring_removes_the_speakers_voice():
    # two speakers far apart in the space; within each, lines differ by mood only
    v = {"a1": [10, 0, 1], "a2": [10, 0, -1], "a3": [10, 0, 0.1], "b1": [0, 10, 1],
         "b2": [0, 10, -1], "b3": [0, 10, 0.1], "c1": [5, 5, 5]}  # fmt: skip
    spk = {k: k[0].upper() for k in v}
    c = emotion.centred(v, spk)
    assert "c1" not in c  # one line: no mean to centre on
    assert emotion.cos(c["a1"], c["b1"]) > 0.9  # the same mood in two voices
    assert emotion.cos(c["a1"], c["a2"]) < -0.5


def test_a_clip_replaces_the_usual_one_only_when_clearly_closer():
    assert emotion.choose({"usual": 1.0, "sad": 0.5}, "usual", 0.25) == "sad"
    assert emotion.choose({"usual": 1.0, "sad": 0.9}, "usual", 0.25) == "usual"
    # the usual clip without a score counts as a typical one (the median)
    assert emotion.choose({"x": 0.2, "y": 1.0, "z": 1.2}, "usual", 0.25) == "x"
    assert emotion.choose({}, "usual", 0.25) == "usual"


def test_top_emotion_skips_other_and_unknown():
    assert emotion.top(_p("unknown", 0.9), LABELS)[0] != "unknown"
    assert emotion.top(_p("sad"), LABELS) == ("sad", 0.8)


def test_agreement_of_the_dubs_emotion_with_the_sources():
    src = {"l1": [1, 0, 0.5], "l2": [1, 0, -0.5], "l3": [1, 0.2, 0], "l4": [1, -0.2, 0]}
    rows_same = [(k, "S1", v, [x + 3 for x in v], _p("sad" if k == "l1" else "neutral"),
                  _p("sad" if k == "l1" else "neutral")) for k, v in src.items()]  # fmt: skip
    a = emotion.agreement(rows_same, LABELS)
    # a voice offset (another language) on every line: centring takes most of it out
    assert a["emotion_cos"] > 0.8 and a["emotion_label_match"] == 1.0
    assert a["emotion_clear_lines"] == 1  # neutral lines are not "clear"
    flat = [(k, "S1", v, [1.0, 0.0, 0.01 * i], _p("sad" if k == "l1" else "neutral"),
             _p("happy")) for i, (k, v) in enumerate(src.items())]  # fmt: skip
    b = emotion.agreement(flat, LABELS)
    assert b["emotion_cos"] < a["emotion_cos"] and b["emotion_label_match"] == 0.0


def test_lines_get_the_clip_of_the_same_emotion():
    from dub.core.pipeline import _line_refs

    units = [Unit(id=f"u{i}", speaker="S1", sentence_ids=[], start_ms=i * 4000,
                  end_ms=i * 4000 + 3000, text="x", n_words=1, asr_conf=0.9)
             for i in range(6)]  # fmt: skip
    # u0 usual (calm); u1 a sobbing clip; u2 an angry clip; u3-u5 lines to voice
    mood = {"u0": [0, 0, 1], "u1": [1, 0, 0], "u2": [0, 1, 0], "u3": [0.9, 0.1, 0],
            "u4": [0.1, 0.9, 0], "u5": [0, 0.1, 0.9]}  # fmt: skip
    centred = emotion.centred({k: v + [5.0] for k, v in mood.items()},
                              {u.id: u.speaker for u in units})  # fmt: skip
    pros = {u.id: [17.0, 3.0, -20.0, 0.8] for u in units}

    def ref(uid: str) -> VoiceRef:
        return VoiceRef(speaker="S1", audio_key=f"k{uid}", text="x", seconds=3.0, unit_id=uid)

    refs = RefBank(refs=[ref("u0")])
    pool = [ref("u0"), ref("u1"), ref("u2")]
    got = _line_refs(units, refs, pool, pros, centred, "emotion", 0.5, 0.25)
    assert got["u3"].unit_id == "u1" and got["u4"].unit_id == "u2"
    assert "u5" not in got  # calm: the usual voice
    # by prosody alone every line sounds the same here: nobody changes voice
    assert _line_refs(units, refs, pool, pros, centred, "prosody", 0.5, 0.25) == {}
