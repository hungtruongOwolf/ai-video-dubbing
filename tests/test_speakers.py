"""Who says each line (M8): phrases of the dialogue and the fusion with the voices."""

from __future__ import annotations

from dub.contracts.v1 import (
    Attribution,
    Character,
    HardsubCue,
    SegmentParams,
    SegmentReq,
    SpeakerLine,
    SpeakerVoice,
    Turn,
    Word,
)
from dub.ops import segment as segment_ops
from dub.ops import speakers as sp


def w(text, s, e, space=True):
    return Word(w=text, start_ms=s, end_ms=e, space_before=space)


def test_transcript_lines_cut_at_pauses_sentence_ends_and_voice_changes():
    words = [
        w("How", 0, 200),
        w("old?", 200, 500),  # sentence end
        w("Three", 600, 900),
        w("months", 900, 1300),
        w("and", 1700, 1900),  # pause of 400 ms before it
        w("a", 1900, 2000),
        w("boy", 2000, 2300),
        w("yes", 2350, 2600),
    ]  # voice change
    turns = [Turn(speaker="S1", start_ms=0, end_ms=2300), Turn(speaker="S2", start_ms=2300,
                                                               end_ms=2600)]  # fmt: skip
    out = sp.lines(words, turns, joiner=" ")
    assert [ln.text for ln in out] == ["How old?", "Three months", "and a boy", "yes"]
    assert [ln.voice for ln in out] == ["S1", "S1", "S1", "S2"]
    assert all(ln.captioned for ln in out)


def test_subtitle_cues_are_the_lines_plus_the_speech_no_cue_covers():
    cues = [
        HardsubCue(start_ms=0, end_ms=1000, text="几个月的", conf=0.9),
        HardsubCue(start_ms=1000, end_ms=1700, text="三个", conf=0.9),
        HardsubCue(start_ms=1700, end_ms=1900, text="三个", conf=0.9),  # same text: merged
        HardsubCue(start_ms=2000, end_ms=2500, text="?", conf=0.3),
    ]  # unsure: dropped
    words = [w("几个月的", 100, 900, False), w("三个", 1000, 1600, False),
             w("啦啦啦", 5000, 6000, False)]  # a song: no cue  # fmt: skip
    turns = [Turn(speaker="S1", start_ms=0, end_ms=7000)]
    out = sp.lines(words, turns, cues, joiner="")
    assert [(ln.text, ln.captioned) for ln in out] == [
        ("几个月的", True),
        ("三个", True),
        ("啦啦啦", False),
    ]
    assert out[1].end_ms == 1900


def attribution(ids, genders):
    return Attribution(
        characters=[Character(id=k, who=k, gender=g) for k, g in genders.items()],
        speaker=ids,
        sure=[True] * len(ids),
    )


def test_fuse_splits_a_quick_exchange_on_one_voice():
    lines = [SpeakerLine(start_ms=i * 1000, end_ms=i * 1000 + 800, text=f"l{i}", voice="S1")
             for i in range(4)]  # fmt: skip
    attr = attribution(["P1", "P2", "P1", "P2"], {"P1": "female", "P2": "female"})
    turns, stats = sp.fuse(lines, attr, [SpeakerVoice(speaker="S1", gender="female",
                                                      gender_conf=0.95)])  # fmt: skip
    assert [t.speaker for t in turns] == ["S1", "S2", "S1", "S2"]
    assert stats["characters"] == 2 and stats["moved"] == 0 and stats["gender_conflicts"] == 0


def test_fuse_refuses_an_attribution_that_changes_the_gender_of_many_lines():
    lines = [SpeakerLine(start_ms=i * 1000, end_ms=i * 1000 + 800, text=f"l{i}",
                         voice="S1" if i % 2 else "S2") for i in range(6)]  # fmt: skip
    voices = [SpeakerVoice(speaker="S1", gender="female", gender_conf=0.95),
              SpeakerVoice(speaker="S2", gender="male", gender_conf=0.95)]  # fmt: skip
    attr = attribution(["P1"] * 6, {"P1": "female"})  # every male line given to the woman
    turns, stats = sp.fuse(lines, attr, voices)
    assert turns is None and stats["rejected"] == "gender" and stats["gender_conflicts"] == 3


def test_a_one_word_answer_survives_segmentation_when_turns_are_lines():
    """Smoothing gives a short run back to the surrounding speaker (an acoustic slip); line
    decisions are not slips, so `smooth=False` keeps the answer."""
    words = [w("Boy", 0, 300), w("or", 300, 500), w("girl?", 500, 900),
             w("Boy.", 1100, 1400),
             w("Any", 1600, 1800), w("milk?", 1800, 2200)]  # fmt: skip
    turns = [Turn(speaker="S1", start_ms=0, end_ms=950), Turn(speaker="S2", start_ms=1050,
             end_ms=1450), Turn(speaker="S1", start_ms=1550, end_ms=2300)]  # fmt: skip
    params = SegmentParams(min_unit_ms=0)

    def speakers(smooth):
        seg = segment_ops.segment(SegmentReq(lang="en", words=words, turns=turns,
                                             params=params, smooth=smooth))  # fmt: skip
        return [u.speaker for u in seg.units]

    assert speakers(False) == ["S1", "S2", "S1"]
    assert set(speakers(True)) == {"S1"}  # the answer went back to the asker
