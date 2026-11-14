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
    # a twin voice (centroids closer than twin_sim) is no competitor: two similar women
    twins = {"S1": [1.0, 0.0, 0.0], "S2": [0.9, 0.436, 0.0]}
    near = [0.8, 0.6, 0.0]  # closer to S2, but S2 sounds like S1
    t = judge(line, _words(line), " ", near, twins["S1"], 0.8, QaLimits(), "S1", twins, 3000)
    assert t.passed and t.closest == "S1"


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
    assert ref.text == "a b c" and 5.0 <= ref.seconds <= 5.05  # u2 (not u0: odd voice)
    # candidates for the audition: the reference first, then the next most typical clips;
    # the clip with another voice in it (u1) is never a candidate while pure ones exist
    many = extract_refs(
        RefsReq(vocals_key=key, units=units, unit_vectors=vectors, purity=purity, candidates=3),
        cas,
    )
    assert many.for_speaker("S1").audio_key == ref.audio_key  # same clip as without audition
    assert [round(r.sim, 2) for r in many.alternates] == [round(ref.sim, 2), 0.32]  # u3, then u0
    assert all(r.centroid == ref.centroid for r in many.alternates)


def test_reference_edges_avoid_a_neighbour_and_short_speakers_are_pooled(tmp_path):
    """Edges move to quiet frames (no neighbour's voice inside the clip); a speaker with only
    short lines gets several of them joined instead of one short clip."""
    import numpy as np

    from dub.ops.audio_io import read_wav, write_wav
    from dub.ops.refs import extract_refs
    from dub.store.cas import CAS

    sr = 48000
    cas = CAS(tmp_path / "cas")
    t = np.arange(sr * 20) / sr
    audio = np.zeros(sr * 20, dtype=np.float32)
    # S1 speaks 2.0-7.0 s; S2 talks right up to 1.85 s (150 ms before S1), then three
    # short lines of 1.2 s at 10, 12 and 14 s
    audio[int(0.5 * sr) : int(1.85 * sr)] = 0.3 * np.sin(2 * np.pi * 300 * t[: int(1.35 * sr)])
    audio[2 * sr : 7 * sr] = 0.2 * np.sin(2 * np.pi * 200 * t[: 5 * sr])
    for s0 in (10, 12, 14):
        audio[s0 * sr : int((s0 + 1.2) * sr)] = 0.2 * np.sin(2 * np.pi * 250 * t[: int(1.2 * sr)])
    wav = tmp_path / "v.wav"
    write_wav(wav, audio, sr)
    key = cas.put_file(wav)

    def u(i, spk, a, b):
        return Unit(id=f"u{i}", speaker=spk, sentence_ids=[], start_ms=a, end_ms=b,
                    text=f"line {i}", n_words=2, asr_conf=0.9)  # fmt: skip

    units = [u(0, "S2", 500, 1850), u(1, "S1", 2000, 7000), u(2, "S2", 10000, 11200),
             u(3, "S2", 12000, 13200), u(4, "S2", 14000, 15200)]  # fmt: skip
    bank = extract_refs(RefsReq(vocals_key=key, units=units), cas)
    s1 = bank.for_speaker("S1")
    clip, csr = read_wav(cas.path(s1.audio_key))
    assert csr == 24000 and 4.95 <= s1.seconds <= 5.25
    lead = clip[: int(0.04 * csr), 0]
    assert np.abs(lead).max() < 0.25  # it does not start inside S2's louder 300 Hz line
    s2 = bank.for_speaker("S2")
    assert s2.seconds >= 3.6 and s2.text.count("line") >= 3  # pooled short lines


def test_text_a_short_clip_cannot_hold_is_dropped():
    from dub.ops.qa import plausible

    heard = _words("cay rồi. hàng trăm triệu tập theo khuynh hướng vô sản trong")
    kept = plausible(heard, 600, "Gầy rồi.", "syllable", " ")
    assert [w.w for w in kept] == ["cay", "rồi."]  # 600 ms holds at most 5.8 syllables
    long_clip = plausible(heard, 3000, "Gầy rồi.", "syllable", " ")
    assert long_clip == heard  # 3 s can hold all of it: a real insertion, judged as such


def test_residual_risks_list_the_lines_worth_a_listen():
    from dub.core.pipeline import residual_risks, risk_summary

    def row(i, risk, reasons, edited=False):
        unit = {"id": f"u{i}", "start_ms": i * 1000, "text": f"line {i}"}
        return {"unit": unit, "placement": {"text": f"dòng {i}"}, "risk": risk,
                "risk_reasons": reasons, "edited": edited}  # fmt: skip

    rows = [row(1, 8.0, ["the dub does not sound like the text (CER 80%)", "very short line"]),
            row(2, 4.0, ["who says it is a guess (the script does not show it)"]),
            row(3, 9.0, ["the dub does not sound like the text (CER 90%)"], edited=True),
            row(4, 0.5, ["very short line"])]  # fmt: skip
    assert risk_summary(rows) == (
        "residual risks: 2 of 4 lines worth a listen (1 words, 1 speaker guessed)"
    )
    out = residual_risks(rows)
    assert out[0] == "## Residual risks" and "`u1` risk 8.0: dòng 1" in out[4]
    assert not any("`u3`" in ln for ln in out)  # a human already looked at it
