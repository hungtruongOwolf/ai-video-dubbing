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


# ---------------------------------------------------------------- M3: boundary probabilities


def test_model_probabilities_split_text_without_punctuation():
    # PhoWhisper-style output: no punctuation, no capitals. Rules see one long sentence.
    ws = words_from("xin chào các bạn hôm nay chúng ta nấu canh cá rất ngon nhé")
    assert len(run(ws).sentences) == 1
    probs = [0.02] * len(ws)
    probs[3] = 0.9  # "...các bạn" | "hôm nay..."
    seg = segment(SegmentReq(lang="vi", words=ws, turns=[], params=SegmentParams(), probs=probs))
    assert [s.text for s in seg.sentences] == [
        "xin chào các bạn",
        "hôm nay chúng ta nấu canh cá rất ngon nhé",
    ]


def test_long_sentence_cuts_follow_the_probabilities():
    ws = words_from(" ".join(f"w{i}" for i in range(40)), step=500)  # 20 s, no punctuation
    probs = [0.02] * 40
    probs[17] = 0.4  # below the sentence threshold, but the best place to cut
    seg = segment(SegmentReq(lang="en", words=ws, turns=[], params=SegmentParams(), probs=probs))
    assert len(seg.sentences) == 1 and len(seg.units) == 2
    assert seg.units[0].text.endswith("w17")


def test_probs_follow_the_words_when_input_is_unsorted():
    ws = words_from("a b c d e f")
    probs = [0.02, 0.02, 0.9, 0.02, 0.02, 0.02]
    shuffled = [ws[i] for i in (5, 2, 0, 4, 1, 3)]
    sprobs = [probs[i] for i in (5, 2, 0, 4, 1, 3)]
    seg = segment(
        SegmentReq(lang="en", words=shuffled, turns=[], params=SegmentParams(), probs=sprobs)
    )
    assert [s.text for s in seg.sentences] == ["a b c", "d e f"]


def test_thai_keeps_the_spaces_the_asr_wrote():
    ws = [
        Word(w="สวัสดี", start_ms=0, end_ms=400),
        Word(w="ครับ", start_ms=400, end_ms=700),
        Word(w="วันนี้", start_ms=900, end_ms=1300, space_before=True),
        Word(w="อากาศดี", start_ms=1300, end_ms=1900),
    ]
    seg = segment(SegmentReq(lang="th", words=ws, turns=[], params=SegmentParams(joiner="")))
    assert seg.units[0].text == "สวัสดีครับ วันนี้อากาศดี"


def test_unit_edges_snap_to_speech_onsets():
    from dub.ops.segment import snap_to_speech

    seg = run(words_from("one two three.") + words_from("four five six.", start_ms=3000))
    u1, u2 = seg.units
    # Whisper put the first unit's start 400 ms early, in the pause (true onset 400 ms later);
    # the voice of the second unit really starts 180 ms before its first word time, and the
    # first unit's voice ends 150 ms after its last word.
    speech = [(u1.start_ms + 400, u1.end_ms + 150), (u2.start_ms - 180, u2.end_ms)]
    s1, s2 = snap_to_speech([u1, u2], speech)
    assert s1.start_ms == u1.start_ms + 400 - 30  # moved LATER, out of the silence
    assert s1.end_ms == u1.end_ms + 150 + 30
    assert s2.start_ms == u2.start_ms - 180 - 30
    # An edge in the middle of long continuous speech stays where it is.
    far = snap_to_speech([u1], [(u1.start_ms - 2000, u1.end_ms + 2000)])
    assert (far[0].start_ms, far[0].end_ms) == (u1.start_ms, u1.end_ms)


def test_a_long_pause_cuts_units_but_not_the_sentence():
    # An orator pauses 2 s (applause) in the middle of a sentence.
    w = words_from("We choose to go to the moon") + words_from(
        "in this decade and do the other things.", start_ms=2800 + 2000
    )
    seg = run(w)
    assert len(seg.sentences) == 1 and len(seg.units) == 2
    assert seg.units[0].sentence_ids == seg.units[1].sentence_ids
    assert seg.units[1].start_ms - seg.units[0].end_ms >= 1500  # no unit spans the pause


def test_punctuation_glued_to_the_next_word_moves_back():
    from dub.ops.segment import normalize_words

    ws = words_from("giữa đời sống lo toan .đặc biệt chùa")
    fixed = [w.w for w in normalize_words(ws)]
    assert fixed == ["giữa", "đời", "sống", "lo", "toan.", "đặc", "biệt", "chùa"]
    seg = run(ws)
    assert [s.text for s in seg.sentences] == ["giữa đời sống lo toan.", "đặc biệt chùa"]


def test_a_stray_word_of_another_speaker_is_given_back():
    from dub.ops.segment import smooth_speakers

    ws = words_from("one two three four five six")
    spk = ["S1", "S1", "S2", "S1", "S1", "S1"]
    ws = [w.model_copy(update={"speaker": s}) for w, s in zip(ws, spk, strict=True)]
    assert [w.speaker for w in smooth_speakers(ws)] == ["S1"] * 6
    long = [w.model_copy(update={"speaker": "S2"}) if 1 <= i <= 4 else w for i, w in enumerate(ws)]
    assert [w.speaker for w in smooth_speakers(long)][1:5] == ["S2"] * 4  # a real turn stays
