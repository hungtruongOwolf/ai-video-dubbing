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
    assert (out[1].start_ms, out[1].end_ms) == (1000, 1600)  # the words, not the picture


def test_a_word_goes_to_the_cue_its_characters_match_not_the_one_on_screen():
    # the reply's cue left the screen 300 ms late: by time, 主 would join the question
    cues = [
        HardsubCue(start_ms=0, end_ms=1500, text="那你啥时候能回来", conf=0.95),
        HardsubCue(start_ms=1500, end_ms=3000, text="主家许我回来", conf=0.95),
    ]
    words = [w(ch, 100 * i, 100 * i + 90, False) for i, ch in enumerate("那你啥时候能回来")]
    words += [w(ch, 1200 + 150 * i, 1200 + 150 * i + 140, False)
              for i, ch in enumerate("主家许我回来")]  # fmt: skip
    turns = [Turn(speaker="S1", start_ms=0, end_ms=3000)]
    out = sp.lines(words, turns, cues, joiner="")
    assert [(ln.text, ln.start_ms) for ln in out] == [
        ("那你啥时候能回来", 0),
        ("主家许我回来", 1200),
    ]
    assert out[0].end_ms == 790


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


def test_songs_stay_original_and_uncaptioned_answers_are_dubbed():
    from dub.contracts.v1 import Unit
    from dub.core.pipeline import _not_dialogue

    def u(uid, spk, s, e):
        return Unit(id=uid, speaker=spk, sentence_ids=["s"], start_ms=s, end_ms=e, text="x",
                    n_words=1)  # fmt: skip

    units = [
        u("a", "S1", 0, 2000),
        u("b", "S2", 3000, 4000),  # "没谁啊": no cue over it
        u("c", "S3", 5000, 8000),
        u("d", "S4", 9000, 10000),
    ]  # a song; a crowd, no cue
    cues = [HardsubCue(start_ms=0, end_ms=2000, text="一"), HardsubCue(start_ms=8500,
            end_ms=8800, text="二")]  # fmt: skip
    hs = {"hardsub_coverage": 0.9}
    # without attribution: every uncaptioned line stays original
    assert _not_dialogue(units, cues, hs, {}) == ["b", "c", "d"]
    st = {"applied": True, "kinds": {"S3": "singing", "S4": "background"}}
    assert _not_dialogue(units, cues, hs, st) == ["c", "d"]
    # no subtitles at all: a song still stays original, a background voice is dubbed
    assert _not_dialogue(units, [], {}, st) == ["c"]


def test_laughs_and_cries_are_not_dubbed():
    # "哈哈" dubbed as "Ha" overran a 20 ms slot and sounded fake: such lines keep the original
    from dub.core.config import load_config

    cfg = load_config(None)
    zh, en, ja, vi = (cfg.langpack(x) for x in ("zh", "en", "ja", "vi"))
    assert all(zh.is_interjection(x) for x in ["哈哈", "啊！", "嗯嗯。", "哎呀", "啊啊啊"])
    assert not any(zh.is_interjection(x) for x in ["好啊", "阿元", "喂", "没谁啊", ""])
    assert all(en.is_interjection(x) for x in ["Haha!", "Uh-huh.", "Oh", "Hmm...", "Wow!"])
    assert not any(en.is_interjection(x) for x in ["Oh no", "Hi", "Yes", "a"])
    assert ja.is_interjection("ふふふ") and not ja.is_interjection("はい")  # はい = yes
    assert not ja.is_interjection("うん")
    assert vi.is_interjection("Haha") and not vi.is_interjection("Ừ")
    assert not cfg.langpack("th").is_interjection("555")  # no pattern: nothing kept


def test_readings_are_voted_line_by_line():
    from dub.contracts.v1 import Attribution, Character
    from dub.ops.speakers import vote

    def reading(ids, chars, sure=None):
        return Attribution(
            characters=[Character(id=c, who=w, kind=k) for c, w, k in chars],
            speaker=ids,
            sure=sure or [True] * len(ids),
        )

    a = reading(["P1", "P2", "P1", "P2", "P3"],
                [("P1", "the nurse", "person"), ("P2", "the steward", "person"),
                 ("P3", "song", "singing")])  # fmt: skip
    # the same reading with other ids (characters numbered differently), and the song
    # seen as a person
    b = reading(["P2", "P1", "P2", "P1", "P3"],
                [("P2", "wet nurse", "person"), ("P1", "Steward Sun", "person"),
                 ("P3", "background song", "singing")])  # fmt: skip
    # a third reading gives line 3 to the steward
    c = reading(["P1", "P2", "P2", "P2", "P3"],
                [("P1", "nurse", "person"), ("P2", "steward", "person"),
                 ("P3", "song", "person")])  # fmt: skip
    voted, agreement = vote([a, b, c])
    assert voted.speaker == ["P1", "P2", "P1", "P2", "P3"]  # 2 of 3 on line 3
    assert voted.sure == [True, True, False, True, True]  # the disagreement is a guess
    assert agreement == 0.8
    assert {ch.id: ch.kind for ch in voted.characters}["P3"] == "singing"  # 2 of 3
    assert vote([a]) == (a, 1.0)
    # a character only another reading found keeps its own (primed) id where it wins
    d = reading(["P1", "P2", "P1", "P4", "P3"],
                [("P1", "n", "person"), ("P2", "s", "person"), ("P4", "boy", "person"),
                 ("P3", "song", "singing")])  # fmt: skip
    e = reading(["P1", "P2", "P1", "P4", "P3"],
                [("P1", "n", "person"), ("P2", "s", "person"), ("P4", "boy", "person"),
                 ("P3", "song", "singing")])  # fmt: skip
    voted, _ = vote([a, d, e])
    assert voted.speaker[3] == "P4'" and "P4'" in {ch.id for ch in voted.characters}
