from __future__ import annotations

from dub.contracts.v1 import RefsReq, Unit, Word
from dub.ops.qa import QaLimits, cosine, judge, tail_heard


def _words(text: str) -> list[Word]:
    return [Word(w=w, start_ms=i * 300, end_ms=i * 300 + 250) for i, w in enumerate(text.split())]


def test_good_take_passes_and_bad_ones_say_why():
    line = "Bà vẫn ăn hết từng thìa cháo nóng"
    vec = [1.0, 0.0, 0.0]
    ok = judge(line, _words("bà vẫn ăn hết từng thìa cháo nóng"), " ", vec, vec, 0.8, QaLimits())
    assert ok.passed and ok.cer == 0 and ok.tail == 1.0 and ok.sim_ratio == 1.25
    cut = judge(line, _words("bà vẫn ăn hết từng"), " ", vec, vec, 0.8, QaLimits())
    assert not cut.passed and any("ending" in r for r in cut.reasons)
    other = judge(line, _words(line.lower()), " ", [0.0, 1.0, 0.0], vec, 0.8, QaLimits())
    assert not other.passed and any("voice" in r for r in other.reasons)
    # sounds like another speaker of the video -> fail, even with a fair similarity ratio
    cents = {"S1": [1.0, 0.0, 0.0], "S2": [0.6, 0.8, 0.0]}
    take = [0.7, 0.71, 0.0]
    swap = judge(line, _words(line), " ", take, cents["S1"], 0.8, QaLimits(), "S1", cents, 3000)
    assert not swap.passed and swap.closest == "S2"
    short = judge(line, _words(line), " ", take, cents["S1"], 0.8, QaLimits(), "S1", cents, 800)
    assert short.passed and short.sim_ratio is None  # too short to judge the voice
    assert ok.score > cut.score and ok.score > other.score


def test_tail_and_cosine():
    assert tail_heard("one two three four", "one two three four") == 1.0
    assert tail_heard("one two three four", "one two") == 0.0
    assert abs(cosine([1, 1], [1, 0]) - 0.7071) < 1e-3


def test_reference_is_the_most_typical_pure_clip(tmp_path):
    import numpy as np

    from dub.ops.audio_io import write_wav
    from dub.ops.refs import extract_refs
    from dub.store.cas import CAS

    cas = CAS(tmp_path / "cas")
    wav = tmp_path / "v.wav"
    write_wav(wav, np.zeros(48000 * 30, dtype=np.float32), 48000)
    key = cas.put_file(wav)
    units = [
        Unit(id=f"u{i}", speaker="S1", sentence_ids=[], start_ms=i * 7000, end_ms=i * 7000 + 5000,
             text="a b c", n_words=3, asr_conf=0.99 if i == 0 else 0.9)
        for i in range(4)
    ]  # fmt: skip
    typical, odd = [1.0, 0.0], [0.0, 1.0]
    vectors = {"u0": odd, "u1": typical, "u2": typical, "u3": typical}
    purity = {"u0": 1.0, "u1": 0.5, "u2": 1.0, "u3": 1.0}  # u1 has another voice in it
    bank = extract_refs(
        RefsReq(vocals_key=key, units=units, unit_vectors=vectors, purity=purity), cas
    )
    ref = bank.for_speaker("S1")
    assert ref.sim > 0.9 and 0 < ref.self_sim < 1 and len(ref.centroid) == 2
    assert ref.text == "a b c" and abs(ref.seconds - 5.12) < 0.01  # u2 (not u0: odd voice)
