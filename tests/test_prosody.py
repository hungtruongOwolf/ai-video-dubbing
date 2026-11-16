"""Voices by delivery: each line voiced from the clip closest to its own (dub.ops.prosody)."""

from __future__ import annotations

import math

import numpy as np

from dub.contracts.v1 import ProsodyReq, RefsReq, Unit
from dub.ops import prosody
from dub.ops.audio_io import write_wav
from dub.store.cas import CAS

SR = 16000


def _voice(f0_from: float, f0_to: float, seconds: float = 1.5, level: float = 0.3) -> np.ndarray:
    """A buzz with harmonics whose pitch glides from f0_from to f0_to (a voiced vowel)."""
    t = np.arange(int(SR * seconds)) / SR
    f0 = f0_from * (f0_to / f0_from) ** (t / seconds)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (level * x / np.abs(x).max()).astype(np.float32)


def test_pitch_is_tracked_in_semitones():
    f = prosody.features(_voice(150, 150))
    assert f is not None
    assert abs(f[0] - 12 * math.log2(150 / 55)) < 0.3  # 17.4 semitones above 55 Hz
    assert f[1] < 0.5 and f[3] > 0.9  # flat, voiced throughout
    glide = prosody.features(_voice(120, 240))
    assert glide is not None and 8 < glide[1] < 12  # an octave, p10-p90
    assert prosody.features(np.zeros(SR, np.float32)) is None  # silence has no pitch


def test_level_follows_the_voice_and_48k_is_read():
    loud, soft = prosody.features(_voice(200, 200)), prosody.features(_voice(200, 200, level=0.03))
    assert loud is not None and soft is not None and abs(loud[2] - soft[2] - 20) < 1
    x48 = np.repeat(_voice(200, 200), 3)  # 48 kHz
    f48 = prosody.features(x48, 48000)
    assert f48 is not None and abs(f48[0] - loud[0]) < 0.5


def test_spans_of_a_file(tmp_path):
    cas = CAS(tmp_path / "cas")
    x = np.concatenate([_voice(110, 110), np.zeros(SR // 2, np.float32), _voice(220, 220)])
    p = tmp_path / "v.wav"
    write_wav(p, x, SR)
    res = prosody.spans(ProsodyReq(audio_key=cas.put_file(p), spans=[(0, 1500), (2000, 3500)]),
                        cas)  # fmt: skip
    low, high = res.features
    assert low is not None and high is not None and abs(high[0] - low[0] - 12) < 0.5


def test_a_line_gets_the_clip_closest_to_its_delivery():
    lines = [[17.0, 3.0, -20.0, 0.8], [17.5, 4.0, -21.0, 0.8], [23.0, 8.0, -12.0, 0.9],
             [12.0, 1.0, -30.0, 0.4]]  # fmt: skip
    mean, sd = prosody.norms(lines)
    pool = [("calm", [17.2, 3.2, -20.5, 0.8]), ("shout", [22.5, 7.5, -13.0, 0.9]),
            ("sob", [12.5, 1.5, -29.0, 0.45])]  # fmt: skip
    assert prosody.pick(lines[2], pool, "calm", mean, sd) == "shout"
    assert prosody.pick(lines[3], pool, "calm", mean, sd) == "sob"
    assert prosody.pick(lines[0], pool, "calm", mean, sd) == "calm"
    # nothing known about the line: the usual reference
    assert prosody.pick(None, pool, "calm", mean, sd) == "calm"
    # a clip barely closer than the usual one (under the margin) does not replace it
    near = [("calm", [17.0, 3.0, -20.0, 0.8]), ("other", [17.6, 3.0, -20.0, 0.8])]
    assert prosody.pick([17.5, 3.0, -20.0, 0.8], near, "calm", mean, sd) == "calm"


def test_agreement_of_the_dub_with_the_source():
    src = [[10.0 + i, 2.0 + i % 3, -20.0 - i, 0.8] for i in range(6)]
    same = [("S1", s, [s[0] + 5, s[1], s[2], s[3]]) for s in src]  # higher voice, same moves
    a = prosody.agreement(same)
    assert a["prosody_f0_corr"] == 1.0 and a["prosody_f0_spread"] == 1.0
    flat = [("S1", s, [15.0 + 0.2 * (i % 2), 2.0, -20.0, 0.8]) for i, s in enumerate(src)]
    b = prosody.agreement(flat)
    assert b["prosody_f0_spread"] < 0.2  # the dub says every line the same way


def test_the_delivery_pool_holds_clean_clips_only(tmp_path):
    from dub.ops.refs import extract_refs

    cas = CAS(tmp_path / "cas")
    wav = tmp_path / "v.wav"
    write_wav(wav, np.zeros(48000 * 40, dtype=np.float32), 48000)
    key = cas.put_file(wav)
    units = [
        Unit(id=f"u{i}", speaker="S1", sentence_ids=[], start_ms=i * 5000,
             end_ms=i * 5000 + (1500 if i == 4 else 3000), text="a b c", n_words=3, asr_conf=0.9)
        for i in range(6)
    ]  # fmt: skip
    vec = {u.id: [1.0, 0.0] for u in units}
    purity = {u.id: 1.0 for u in units} | {"u1": 0.6}  # u1 has another voice in it
    bank = extract_refs(RefsReq(vocals_key=key, units=units, unit_vectors=vec, purity=purity,
                                styles=12), cas)  # fmt: skip
    ids = {r.unit_id for r in bank.styles}
    assert ids == {"u0", "u2", "u3", "u5"}  # not u1 (impure), not u4 (1.5 s, too short)
    assert bank.for_speaker("S1").unit_id in ids
    assert not extract_refs(RefsReq(vocals_key=key, units=units, unit_vectors=vec,
                                    purity=purity), cas).styles  # fmt: skip


def test_the_delivery_pool_survives_the_reference_transcripts():
    import asyncio

    from dub.contracts.v1 import AudioRes, RefBank, VoiceRef, Word, WordTimeline
    from dub.core.pipeline import _retranscribe_refs

    class Cx:  # what _retranscribe_refs asks of the job context
        async def local(self, op, fn, req, res_type, subject=None):
            return AudioRes(audio_key=req.audio_key, dur_ms=1000)

        async def call(self, op, req, subject=None):
            return WordTimeline(lang="zh", words=[Word(w="a", start_ms=0, end_ms=100)])

    ref = VoiceRef(speaker="S1", audio_key="k1", text="a", seconds=3.0, unit_id="u1")
    pool = [ref.model_copy(update={"audio_key": "k2", "unit_id": "u2"})]
    out = asyncio.run(_retranscribe_refs(Cx(), RefBank(refs=[ref], styles=pool), "zh", "",
                                         lambda m: None))  # fmt: skip
    assert out.styles == pool  # the first run of M12 lost them here: no line got its delivery


def test_a_line_next_to_another_speakers_is_cut_inside_its_edges(tmp_path):
    from dub.ops.refs import extract_refs

    cas = CAS(tmp_path / "cas")
    sr = 48000
    x = np.zeros(sr * 12, dtype=np.float32)
    x[int(3.9 * sr) : int(4.0 * sr)] = 0.5  # the other speaker, 100 ms before u1
    wav = tmp_path / "v.wav"
    write_wav(wav, x, sr)
    key = cas.put_file(wav)
    units = [
        Unit(id="u0", speaker="S2", sentence_ids=[], start_ms=3800, end_ms=3990, text="x",
             n_words=1, asr_conf=0.9),
        Unit(id="u1", speaker="S1", sentence_ids=[], start_ms=4100, end_ms=7100, text="a b c",
             n_words=3, asr_conf=0.9),
    ]  # fmt: skip
    bank = extract_refs(RefsReq(vocals_key=key, units=units, styles=12), cas)
    s1 = [r for r in bank.styles if r.speaker == "S1"]
    assert len(s1) == 1  # kept in the pool
    from dub.ops.audio_io import read_wav

    clip, _ = read_wav(cas.path(s1[0].audio_key))
    assert np.abs(clip).max() < 0.01  # and nothing of S2's voice, 100 ms before it, in it
