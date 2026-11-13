"""Mix the dubbed lines over the bed (music + effects), then normalise loudness."""

from __future__ import annotations

import numpy as np

from dub.contracts.v1 import AudioRes, MixReq
from dub.ops.audio_io import duration_ms, read_wav, write_wav
from dub.ops.media import loudnorm
from dub.store.cas import CAS

SR = 48000
FADE_MS = 8  # short fade in/out so line starts and ends do not click


def _fade(clip: np.ndarray, sr: int) -> np.ndarray:
    n = min(len(clip) // 2, int(sr * FADE_MS / 1000))
    if n <= 0:
        return clip
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    clip = clip.copy()
    clip[:n] *= ramp
    clip[-n:] *= ramp[::-1]
    return clip


def mix(req: MixReq, cas: CAS) -> AudioRes:
    bed, sr = read_wav(cas.path(req.bed_key))
    if sr != SR:
        raise ValueError(f"The bed must be 48 kHz, got {sr}")
    n_total = max(len(bed), int(req.duration_ms * SR / 1000))
    out = np.zeros((n_total, 2), dtype=np.float32)
    out[: len(bed)] = bed[:, :2] if bed.shape[1] >= 2 else np.repeat(bed, 2, axis=1)
    out *= 10 ** (req.bed_gain_db / 20)

    for p in req.placements:
        if not p.audio_key:
            continue
        clip, csr = read_wav(cas.path(p.audio_key))
        if csr != SR:
            raise ValueError(f"Clip {p.unit_id} must be 48 kHz, got {csr}")
        mono = _fade(clip.mean(axis=1), SR)
        start = int(p.start_ms * SR / 1000)
        end = min(n_total, start + len(mono))
        if end > start:
            out[start:end] += mono[: end - start, None]

    # Soft limiter: never exceeds 1.0 but leaves quiet signals almost untouched.
    peak = float(np.abs(out).max(initial=0.0))
    if peak > 0.95:
        out = np.tanh(out / peak * 1.5) / np.tanh(1.5) * 0.95

    raw = cas.tmp_path(".wav")
    write_wav(raw, out, SR)
    final = cas.tmp_path(".wav")
    loudnorm(raw, final, req.loudness_lufs)
    raw.unlink(missing_ok=True)
    dur = duration_ms(final)
    return AudioRes(audio_key=cas.put_file(final, move=True), dur_ms=dur)
