from __future__ import annotations

from dub.contracts.v1 import SegmentParams, SegmentReq, Turn, Word
from dub.demo import build_words
from dub.ops.segment import segment, split_long


def words_from(text: str, start_ms: int = 0, step: int = 400, speaker: str = "S1") -> list[Word]:
    out = []
    t = start_ms
    for tok in text.split():
        out.append(Word(w=tok, start_ms=t, end_ms=t + step - 50, speaker=speaker))
        t += step
    return out


def run(words, turns=None, **params):
    return segment(
        SegmentReq(lang="en", words=words, turns=turns or [], params=SegmentParams(**params))
    )


def test_sentences_split_on_terminal_punctuation_not_abbreviations():
    seg = run(words_from("Hello there. I met Mr. Tu and Dr. Lee today! Fine?"))
    texts = [s.text for s in seg.sentences]
    assert texts == ["Hello there.", "I met Mr. Tu and Dr. Lee today!", "Fine?"]


def test_speaker_change_is_a_hard_boundary():
    w = words_from("one two three four", speaker="S1") + words_from(
        "five six seven eight", start_ms=1600, speaker="S2"
    )
    turns = [
        Turn(speaker="S1", start_ms=0, end_ms=1550),
        Turn(speaker="S2", start_ms=1600, end_ms=3200),
    ]
    seg = run(w, turns)
    assert [u.speaker for u in seg.units] == ["S1", "S2"]
    assert seg.units[0].end_ms <= seg.units[1].start_ms


def test_long_pause_is_a_hard_boundary():
    w = words_from("a b c d") + words_from("e f g h", start_ms=1600 + 2000)
    seg = run(w)
    assert len(seg.units) == 2


def test_no_unit_longer_than_max_and_cuts_prefer_commas():
    text = (
        "this is a very long sentence that keeps going, and going without any stop, "
        "because the speaker never breathes, and we must cut it somewhere sensible."
    )
    ws = words_from(text, step=650)  # ~17 s: longer than the 15 s limit
    p = SegmentParams()
    chunks = split_long(ws, p)
    assert len(chunks) >= 2
    for c in chunks:
        assert c[-1].end_ms - c[0].start_ms <= p.max_unit_ms
    # every cut lands after a comma
    assert all(c[-1].w.endswith((",", ".")) for c in chunks)


def test_tiny_units_are_merged():
    seg = run(words_from("Oh no! That is so sweet.", step=300))
    assert len(seg.units) == 1
    assert seg.units[0].text == "Oh no! That is so sweet."
    assert len(seg.units[0].sentence_ids) == 2


def test_demo_script_respects_limits():
    seg = run(build_words())
    assert all(u.slot_ms <= 15000 for u in seg.units)
    assert all(u.slot_ms >= 1000 or u.n_words <= 3 for u in seg.units)
    assert {u.speaker for u in seg.units} == {"S1", "S2"}


def test_deterministic_ids():
    a = run(words_from("Hello there. General Kenobi."))
    b = run(words_from("Hello there. General Kenobi."))
    assert [u.id for u in a.units] == [u.id for u in b.units]
