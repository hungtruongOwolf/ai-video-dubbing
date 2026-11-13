"""Synthetic test signals: a buzzy "voice" and a soft music pad.

Used by the fake TTS provider and by `dub demo` to build a test video. Not a real TTS.
"""

from __future__ import annotations

import numpy as np


def voice(dur_ms: int, f0: float, seed: int, sr: int = 48000) -> np.ndarray:
    """A voice-like buzz: harmonics + vibrato + a syllable-rate amplitude envelope."""
    n = max(1, int(sr * dur_ms / 1000))
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(seed)
    vib = 1.0 + 0.03 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * f0 * np.cumsum(vib) / sr
    sig = sum(a * np.sin(k * phase) for k, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)))
    env = 0.35 + 0.65 * np.sin(np.pi * 4.5 * t + rng.uniform(0, np.pi)) ** 2
    out = (sig * env * 0.16).astype(np.float32)
    fade = min(n // 2, int(sr * 0.01))
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out


def pad(dur_ms: int, sr: int = 48000, level: float = 0.03) -> np.ndarray:
    """A quiet A-minor chord with slow tremolo, standing in for background music."""
    n = int(sr * dur_ms / 1000)
    t = np.arange(n, dtype=np.float64) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 261.63, 329.63))
    trem = 0.7 + 0.3 * np.sin(2 * np.pi * 0.25 * t)
    return (chord / 3 * trem * level).astype(np.float32)
