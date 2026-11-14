"""Per-line loudness: each dubbed line follows the level of the original line."""

from __future__ import annotations

import numpy as np

from dub.contracts.v1 import LineLevelReq, Placement, TileReq
from dub.ops.audio_io import read_wav, write_wav
from dub.ops.level import active_level_db, gains_for, level_metrics, line_level
from dub.ops.mix import mix_tile
from dub.store.cas import CAS

SR = 48000


def _tone(seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def _put(cas: CAS, tmp_path, name: str, mono: np.ndarray) -> str:
    f = tmp_path / name
    write_wav(f, np.stack([mono, mono], axis=1), SR)
    return cas.put_file(f)


def _db(amp: float) -> float:
    return 20 * np.log10(amp / np.sqrt(2))  # sine power


def test_active_level_ignores_pauses():
    voice = _tone(1.0, 0.1)
    with_pause = np.concatenate([voice, np.zeros(SR * 2, dtype=np.float32), voice])
    assert abs(active_level_db(voice, SR) - _db(0.1)) < 0.1
    # two seconds of silence in the middle do not make the line quieter
    assert abs(active_level_db(with_pause, SR) - active_level_db(voice, SR)) < 0.1
    assert active_level_db(np.zeros(SR, dtype=np.float32), SR) is None
    assert active_level_db(_tone(0.05, 0.1), SR) is None  # too short to judge


def _placement(uid: str, spk: str, slot: tuple[int, int], key: str) -> Placement:
    a, b = slot
    return Placement(
        unit_id=uid, speaker=spk, slot_start_ms=a, slot_end_ms=b, start_ms=a, end_ms=b,
        natural_ms=b - a, clip_ms=b - a, rate=1.0, variant="target", text="x", audio_key=key,
    )  # fmt: skip


def _levels(cas, tmp_path, vocals, clips, slots):
    vkey = _put(cas, tmp_path, "vocals.wav", vocals)
    out = {}
    for i, (clip, (a, b)) in enumerate(zip(clips, slots, strict=True)):
        key = _put(cas, tmp_path, f"c{i}.wav", clip)
        req = LineLevelReq(vocals_key=vkey, slot_start_ms=a, slot_end_ms=b, audio_key=key)
        out[f"u{i}"] = line_level(req, cas)
    return out


def test_lines_follow_the_original_levels(tmp_path):
    cas = CAS(tmp_path / "cas")
    # original: a normal line, a shout (+12 dB), an aside (-8 dB), then a slot with no voice
    levels = [0.05, 0.05 * 4, 0.05 * 0.4, 0.0]
    vocals = np.concatenate([_tone(2.0, a) for a in levels])
    # the TTS returned every line at the same level
    lv = _levels(
        cas,
        tmp_path,
        vocals,
        [_tone(1.5, 0.2)] * 4,
        [(i * 2000, i * 2000 + 2000) for i in range(4)],
    )
    g = gains_for(lv, {f"u{i}": "S1" for i in range(4)}, max_dev_db=10.0)
    assert lv["u3"].src_db is None
    median = np.median([lv[u].src_db for u in ("u0", "u1", "u2")])
    assert abs(median - _db(0.05)) < 0.1
    # relative levels: the shout is +12 dB, but at most max_dev (10) above the speaker median
    assert abs((g["u1"] - g["u0"]) - 10.0) < 0.1
    assert abs((g["u2"] - g["u0"]) - 20 * np.log10(0.4)) < 0.1
    assert abs(g["u0"] - (_db(0.05) - _db(0.2))) < 0.1  # -12 dB: TTS was louder
    assert abs(g["u3"] - g["u0"]) < 0.01  # unmeasurable original: the speaker's median
    m = level_metrics(lv, g)
    assert abs(m["level_offset_db_p50"] - 12.04) < 0.1
    assert m["level_corr_raw"] == 0.0  # the raw TTS levels did not follow the original at all


def test_gain_is_capped(tmp_path):
    cas = CAS(tmp_path / "cas")
    lv = _levels(cas, tmp_path, _tone(2.0, 0.5), [_tone(1.0, 0.005)], [(0, 2000)])  # 40 dB
    assert gains_for(lv, {"u0": "S"}, max_gain_db=15)["u0"] == 15.0


def test_mix_applies_the_line_gain(tmp_path):
    cas = CAS(tmp_path / "cas")
    bed = _put(cas, tmp_path, "bed.wav", np.zeros(SR * 2, dtype=np.float32))
    clip = _put(cas, tmp_path, "c.wav", _tone(1.0, 0.1))
    p = _placement("u", "S", (0, 1000), clip)
    loud = p.model_copy(update={"gain_db": 6.0})
    outs = []
    for q in (p, loud):
        tile = mix_tile(TileReq(bed_key=bed, start_ms=0, end_ms=2000, placements=[q]), cas)
        outs.append(read_wav(cas.path(tile.audio_key))[0][:, 0])
    ratio = np.abs(outs[1][SR // 4 : SR // 2]).max() / np.abs(outs[0][SR // 4 : SR // 2]).max()
    assert abs(20 * np.log10(ratio) - 6.0) < 0.1
