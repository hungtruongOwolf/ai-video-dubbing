"""Per-line loudness: each dubbed line follows the level of the original line.

The TTS returns every line at roughly the level of its reference clip, whatever the original
did: a shout and an aside come out alike, and the balance between the voice and the music
bed depends on the engine instead of the film. Here each line gets the gain that brings its
active speech level to the level of the original voice in the same slot (the separated
vocals), within limits:

- the original level is taken relative to the speaker's median, and at most `max_dev_db`
  away from it: a slot full of bleed (applause, music the separator left in the vocals)
  cannot make one line jump out;
- a line whose original level cannot be measured gets the speaker's median;
- the gain itself is capped at `max_gain_db` either way.

"Active level" is the mean power of the 20 ms frames that carry speech: frames below an
absolute gate (-55 dBFS) or more than 25 dB under the line's loud frames (p95) are pauses
and do not count, so a line with a long pause is not judged quieter than it sounds.
"""

from __future__ import annotations

import numpy as np

from dub.contracts.v1 import LevelReq, LevelRes
from dub.ops.audio_io import read_wav, read_wav_range
from dub.store.cas import CAS

SR = 48000
FRAME_MS = 20
ABS_GATE_DB = -55.0
REL_GATE_DB = -25.0
MIN_ACTIVE_FRAMES = 5  # 100 ms of speech


def active_level_db(x: np.ndarray, sr: int) -> float | None:
    """Mean power (dBFS) of the active frames, or None when almost nothing is active."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    n = max(1, int(sr * FRAME_MS / 1000))
    k = len(mono) // n
    if k < MIN_ACTIVE_FRAMES:
        return None
    power = (mono[: k * n].astype(np.float64).reshape(k, n) ** 2).mean(axis=1)
    db = 10 * np.log10(power + 1e-12)
    gate = max(ABS_GATE_DB, float(np.percentile(db, 95)) + REL_GATE_DB)
    active = power[db > gate]
    if len(active) < MIN_ACTIVE_FRAMES:
        return None
    return round(float(10 * np.log10(active.mean())), 2)


def match_levels(req: LevelReq, cas: CAS) -> LevelRes:
    src: dict[str, float | None] = {}
    dub: dict[str, float | None] = {}
    vocals = cas.path(req.vocals_key)
    for p in req.placements:
        if not p.audio_key:
            continue
        clip, csr = read_wav(cas.path(p.audio_key))
        dub[p.unit_id] = active_level_db(clip, csr)
        # the original line: its own slot of the separated vocals (48 kHz, like the bed)
        a = int(p.slot_start_ms * SR / 1000)
        seg, sr = read_wav_range(vocals, a, int(p.slot_end_ms * SR / 1000) - a)
        if sr != SR:
            raise ValueError(f"The vocals must be {SR} Hz, got {sr}")
        src[p.unit_id] = active_level_db(seg, sr)

    by_spk: dict[str, list[float]] = {}
    for p in req.placements:
        v = src.get(p.unit_id)
        if v is not None:
            by_spk.setdefault(p.speaker, []).append(v)
    median = {s: float(np.median(v)) for s, v in by_spk.items()}

    gains: dict[str, float] = {}
    for p in req.placements:
        d = dub.get(p.unit_id)
        m = median.get(p.speaker)
        if d is None or m is None:
            gains[p.unit_id] = 0.0
            continue
        s = src.get(p.unit_id)
        target = m if s is None else m + max(-req.max_dev_db, min(req.max_dev_db, s - m))
        gains[p.unit_id] = round(max(-req.max_gain_db, min(req.max_gain_db, target - d)), 2)
    return LevelRes(gains_db=gains, src_db=src, dub_db=dub)


def level_metrics(res: LevelRes) -> dict[str, float | int]:
    """How far the raw TTS levels were from the original, line by line (before the gains).

    offset: dub - original, per line. Its median is the overall offset (one gain would fix
    it); its spread (p90 of the distance to that median) is what a single gain cannot fix:
    lines louder or quieter than they were in the original, relative to each other.
    """
    pairs = [
        (res.dub_db[u], res.src_db[u])
        for u in res.gains_db
        if res.dub_db.get(u) is not None and res.src_db.get(u) is not None
    ]
    if len(pairs) < 3:
        return {}
    d = np.array([a for a, _ in pairs])
    s = np.array([b for _, b in pairs])
    off = d - s
    med = float(np.median(off))
    gains = np.abs(np.array(list(res.gains_db.values())))
    corr = float(np.corrcoef(d, s)[0, 1]) if d.std() > 0 and s.std() > 0 else 0.0
    return {
        "level_offset_db_p50": round(med, 2),
        "level_mismatch_db_p90": round(float(np.percentile(np.abs(off - med), 90)), 2),
        "level_corr_raw": round(corr, 3),
        "level_gain_db_p95": round(float(np.percentile(gains, 95)), 2),
    }
