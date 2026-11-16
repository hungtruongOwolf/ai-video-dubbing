"""Prosody of a line: how high, how wide and how loud it is said (M12).

Voice-cloning TTS copies the delivery of its reference clip along with the timbre. With one
reference per speaker, every line of that speaker is said in the mood of that clip. In the
drama the clip was a grieving line, and the dub's line-to-line pitch spread was 1/3 to 1/6 of
the source's (M10 review). So each line gets the reference whose delivery is closest to its
own (`pick`), out of the speaker's clean lines (refs.extract_refs, `styles`). `agreement`
measures whether the dub's lines follow the source's lines.

Features of a clip (`features`), from the separated vocals:
- f0: median pitch, in semitones above 55 Hz;
- f0_range: p90 - p10 of the pitch, in semitones (flat vs lively);
- energy: mean level of the active frames, dB (soft vs shouted);
- voiced: share of the active frames that are voiced (breathy, sobbing, whispered: low).
Pitch is tracked with YIN (de Cheveigné and Kawahara 2002), vectorised over frames: 40 ms
windows every 10 ms, 70-500 Hz.
"""

from __future__ import annotations

import math
import wave

import numpy as np

from dub.contracts.v1 import ProsodyReq, ProsodyRes
from dub.ops.audio_io import read_wav, read_wav_range, resample_int
from dub.store.cas import CAS

SR = 16000
WIN = 640  # 40 ms
HOP = 160  # 10 ms
FMIN, FMAX = 70, 500
YIN_THRESHOLD = 0.15
VOICED_MAX = 0.3  # aperiodicity of a voiced frame
ACTIVE_DB = 35.0  # frames within this of the clip's loudest are active
FLOOR_DB = -60.0  # ...and never quieter than this (silence has no pitch)
MIN_VOICED = 10  # frames: fewer and the clip has no pitch to speak of
FEATURES = ("f0", "f0_range", "energy", "voiced")
WEIGHTS = (1.0, 0.5, 1.0, 0.5)
FLOOR = (1.0, 1.0, 2.0, 0.08)  # the least spread a speaker's feature is given (z-scores)


def track(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """(f0 in Hz, NaN where unvoiced; level in dB) per 10 ms frame of a 16 kHz signal."""
    tmax, tmin = sr // FMIN, sr // FMAX
    n = 1 + (len(x) - WIN) // HOP
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx].astype(np.float64)
    fr -= fr.mean(axis=1, keepdims=True)
    db = 10 * np.log10((fr**2).mean(axis=1) + 1e-12)
    w = WIN - tmax
    nfft = 1 << (WIN + w - 1).bit_length()
    r = np.fft.irfft(np.conj(np.fft.rfft(fr[:, :w], nfft)) * np.fft.rfft(fr, nfft), nfft)
    r = r[:, : tmax + 1]
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(fr**2, axis=1)], axis=1)
    taus = np.arange(tmax + 1)
    e0 = cs[:, w][:, None]
    et = cs[:, taus + w] - cs[:, taus]
    d = e0 + et - 2 * r
    d[:, 0] = 0
    cum = np.cumsum(d[:, 1:], axis=1)
    cmndf = np.ones_like(d)
    cmndf[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cum, 1e-12)
    band = cmndf[:, tmin : tmax + 1]
    below = band < YIN_THRESHOLD
    # the first dip under the threshold (YIN step 4), else the deepest one
    first = np.where(below.any(axis=1), below.argmax(axis=1), band.argmin(axis=1))
    # walk down to the bottom of that dip
    for _ in range(4):
        nxt = np.minimum(first + 1, band.shape[1] - 1)
        step = band[np.arange(n), nxt] < band[np.arange(n), first]
        first = np.where(step, nxt, first)
    ap = band[np.arange(n), first]
    tau = (first + tmin).astype(np.float64)
    f0 = sr / tau
    active = db > max(db.max() - ACTIVE_DB, FLOOR_DB)
    f0[(ap > VOICED_MAX) | ~active] = np.nan
    return f0, db


def features(x: np.ndarray, sr: int = SR) -> list[float] | None:
    """[f0, f0_range, energy, voiced] of a clip, or None when it holds too little voice."""
    if sr != SR:
        if sr % SR:
            raise ValueError(f"prosody needs a multiple of {SR} Hz, got {sr}")
        x = resample_int(x, 1, sr // SR)
    f0, db = track(x)
    if len(db) == 0:
        return None
    active = db > max(db.max() - ACTIVE_DB, FLOOR_DB)
    voiced = ~np.isnan(f0)
    if voiced.sum() < MIN_VOICED:
        return None
    st = 12 * np.log2(f0[voiced] / 55.0)
    return [
        round(float(np.median(st)), 2),
        round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2),
        round(float(db[active].mean()), 2),
        round(float(voiced.sum() / max(1, active.sum())), 3),
    ]


def spans(req: ProsodyReq, cas: CAS) -> ProsodyRes:
    """Features of each span of a WAV (the separated vocals), or of the whole file."""
    path = cas.path(req.audio_key)
    if not req.spans:
        audio, sr = read_wav(path)
        return ProsodyRes(features=[features(audio.mean(axis=1), sr)])
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
    out: list[list[float] | None] = []
    for a, b in req.spans:
        audio, sr = read_wav_range(path, a * rate // 1000, max(0, (b - a) * rate // 1000))
        out.append(features(audio.mean(axis=1), sr) if len(audio) else None)
    return ProsodyRes(features=out)


def norms(feats: list[list[float]]) -> tuple[list[float], list[float]]:
    """Mean and spread of a speaker's features (spread floored: FLOOR)."""
    cols = list(zip(*feats, strict=True))
    mean = [sum(c) / len(c) for c in cols]
    sd = [
        max(math.sqrt(sum((v - m) ** 2 for v in c) / max(1, len(c) - 1)), f)
        for c, m, f in zip(cols, mean, FLOOR, strict=True)
    ]
    return mean, sd


def distance(a: list[float], b: list[float], sd: list[float]) -> float:
    return math.sqrt(
        sum(w * ((x - y) / s) ** 2 for x, y, s, w in zip(a, b, sd, WEIGHTS, strict=True))
    )


def pick(
    line: list[float] | None,
    pool: list[tuple[str, list[float]]],
    default: str,
    mean: list[float],
    sd: list[float],
    margin: float = 0.5,
) -> str:
    """The reference whose delivery is closest to the line's: a pool clip only when it is
    closer than the speaker's usual reference by `margin` (z units). The usual reference is
    the speaker's most typical clip; without features of its own it counts as the speaker's
    mean delivery."""
    if line is None or not pool:
        return default
    by = dict(pool)
    d_default = distance(line, by.get(default, mean), sd)
    key, feats = min(pool, key=lambda kv: distance(line, kv[1], sd))
    return key if distance(line, feats, sd) < d_default - margin else default


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (sx * sy)


def agreement(rows: list[tuple[str, list[float], list[float]]]) -> dict[str, float]:
    """Does the dub follow the source from line to line? rows: (speaker, source features,
    dub features). Each speaker's lines are centred on that speaker's mean (a voice is higher
    or lower as a whole in another language), then pooled: the correlation of the dub's pitch,
    pitch range and level with the source's, and the dub's pitch spread over the source's."""
    by: dict[str, list[tuple[list[float], list[float]]]] = {}
    for spk, s, d in rows:
        by.setdefault(spk, []).append((s, d))
    cen: dict[int, tuple[list[float], list[float]]] = {i: ([], []) for i in range(3)}
    spread_s, spread_d = [], []
    for pairs in by.values():
        if len(pairs) < 3:
            continue
        for i in range(3):
            ms = sum(s[i] for s, _ in pairs) / len(pairs)
            md = sum(d[i] for _, d in pairs) / len(pairs)
            cen[i][0].extend(s[i] - ms for s, _ in pairs)
            cen[i][1].extend(d[i] - md for _, d in pairs)
        spread_s.extend(s[0] - sum(p[0] for p, _ in pairs) / len(pairs) for s, _ in pairs)
        spread_d.extend(d[0] - sum(p[0] for _, p in pairs) / len(pairs) for _, d in pairs)
    out: dict[str, float] = {}
    for i, name in enumerate(("f0", "f0_range", "energy")):
        c = _corr(*cen[i])
        if c is not None:
            out[f"prosody_{name}_corr"] = round(c, 3)
    if len(spread_s) >= 3:
        ss = math.sqrt(sum(v * v for v in spread_s) / len(spread_s))
        sd = math.sqrt(sum(v * v for v in spread_d) / len(spread_d))
        if ss > 0:
            out["prosody_f0_spread"] = round(sd / ss, 3)
    out["prosody_lines"] = len(rows)
    return out
