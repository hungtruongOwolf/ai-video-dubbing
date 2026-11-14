"""Read/write 16-bit PCM WAV with the stdlib `wave` module + numpy (no libsndfile needed)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (float32 array of shape [n, channels] in [-1, 1], sample rate)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"Only 16-bit WAV is supported, got {8 * width}-bit: {path}")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return round(w.getnframes() * 1000 / w.getframerate())


def read_wav_range(path: Path, start_frame: int, n_frames: int) -> tuple[np.ndarray, int]:
    """Read only [start_frame, start_frame + n_frames) of a WAV; short at the end of the file."""
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        if w.getsampwidth() != 2:
            raise ValueError(f"Only 16-bit WAV is supported: {path}")
        start = min(max(0, start_frame), w.getnframes())
        w.setpos(start)
        raw = w.readframes(max(0, n_frames))
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr
