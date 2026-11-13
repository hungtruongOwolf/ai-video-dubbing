"""Tiny energy-based voice activity detection (VAD). Good enough for fakes and tests.

The real pipeline uses Silero VAD (ONNX, CPU) from M1.
"""

from __future__ import annotations

import numpy as np


def energy_vad(
    mono: np.ndarray,
    sr: int,
    *,
    frame_ms: int = 30,
    min_speech_ms: int = 200,
    merge_gap_ms: int = 300,
    rel_db: float = 12.0,
    floor_db: float = -45.0,
) -> list[tuple[int, int]]:
    """Return speech regions as (start_ms, end_ms)."""
    hop = max(1, int(sr * frame_ms / 1000))
    n = len(mono) // hop
    if n == 0:
        return []
    frames = mono[: n * hop].reshape(n, hop)
    rms_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    noise = np.percentile(rms_db, 10)
    thr = max(floor_db, noise + rel_db)
    active = rms_db > thr

    regions: list[tuple[int, int]] = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            regions.append((start * frame_ms, i * frame_ms))
            start = None
    if start is not None:
        regions.append((start * frame_ms, n * frame_ms))

    merged: list[tuple[int, int]] = []
    for s, e in regions:
        if merged and s - merged[-1][1] <= merge_gap_ms:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_speech_ms]
