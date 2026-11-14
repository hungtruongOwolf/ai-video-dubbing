"""Per-line loudness: each dubbed line follows the level of the original line.

The TTS returns every line at roughly the level of its reference clip, whatever the original
did (OmniVoice scales its output to the reference RMS): a shout and an aside come out alike,
and the balance between the voice and the music bed depends on the engine instead of the
film. Here each line gets the gain that brings its active speech level to the level of the
original voice in the same slot (the separated vocals), within limits:

- the original level is taken relative to the speaker's median, and at most `max_dev_db`
  away from it: a slot full of bleed (applause, music the separator left in the vocals)
  cannot make one line jump out;
- a line whose original level cannot be measured gets the speaker's median;
- the gain itself is capped at `max_gain_db` either way.

"Active level" is the mean power of the 20 ms frames that carry speech: frames below an
absolute gate (-55 dBFS) or more than 25 dB under the line's loud frames (p95) are pauses
and do not count, so a line with a long pause is not judged quieter than it sounds.

Each line is measured as soon as it is voiced (`line_level`, cached per line, off the
critical path); the gains need every line of the speaker and are computed at the end.
"""

from __future__ import annotations

import numpy as np

from dub.contracts.v1 import LineLevel, LineLevelReq
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


def line_level(req: LineLevelReq, cas: CAS) -> LineLevel:
    """Active level of one dubbed clip and of the original line in its slot."""
    clip, csr = read_wav(cas.path(req.audio_key))
    a = int(req.slot_start_ms * SR / 1000)
    seg, sr = read_wav_range(cas.path(req.vocals_key), a, int(req.slot_end_ms * SR / 1000) - a)
    if sr != SR:
        raise ValueError(f"The vocals must be {SR} Hz, got {sr}")
    return LineLevel(src_db=active_level_db(seg, sr), dub_db=active_level_db(clip, csr))


def gains_for(
    levels: dict[str, LineLevel],
    speakers: dict[str, str],
    max_dev_db: float = 10.0,
    max_gain_db: float = 24.0,
) -> dict[str, float]:
    """unit id -> gain (dB) that puts the line at the original's level (see the module doc)."""
    by_spk: dict[str, list[float]] = {}
    for uid, lv in levels.items():
        if lv.src_db is not None and uid in speakers:
            by_spk.setdefault(speakers[uid], []).append(lv.src_db)
    median = {s: float(np.median(v)) for s, v in by_spk.items()}
    gains: dict[str, float] = {}
    for uid, lv in levels.items():
        m = median.get(speakers.get(uid, ""))
        if lv.dub_db is None or m is None:
            gains[uid] = 0.0
            continue
        s = lv.src_db
        target = m if s is None else m + max(-max_dev_db, min(max_dev_db, s - m))
        gains[uid] = round(max(-max_gain_db, min(max_gain_db, target - lv.dub_db)), 2)
    return gains


def level_metrics(levels: dict[str, LineLevel], gains: dict[str, float]) -> dict[str, float]:
    """How far the raw TTS levels were from the original, line by line (before the gains).

    offset: dub - original, per line. Its median is the overall offset (one gain would fix
    it); its spread (p90 of the distance to that median) is what a single gain cannot fix:
    lines louder or quieter than they were in the original, relative to each other.
    """
    pairs = [
        (lv.dub_db, lv.src_db)
        for lv in levels.values()
        if lv.dub_db is not None and lv.src_db is not None
    ]
    if len(pairs) < 3:
        return {}
    d = np.array([a for a, _ in pairs])
    s = np.array([b for _, b in pairs])
    off = d - s
    med = float(np.median(off))
    g = np.abs(np.array(list(gains.values()) or [0.0]))
    corr = float(np.corrcoef(d, s)[0, 1]) if d.std() > 0 and s.std() > 0 else 0.0
    return {
        "level_offset_db_p50": round(med, 2),
        "level_mismatch_db_p90": round(float(np.percentile(np.abs(off - med), 90)), 2),
        "level_corr_raw": round(corr, 3),
        "level_gain_db_p95": round(float(np.percentile(g, 95)), 2),
    }
