"""Short MP3 previews for the review page: a few seconds of the dub (from the mixed tiles) or
of the original audio, around one line."""

from __future__ import annotations

import numpy as np

from dub.ops.audio_io import read_wav_range, write_wav
from dub.ops.media import run_ffmpeg
from dub.store.cas import CAS


def cut_mp3(cas: CAS, segments: list[tuple[str, int]], from_ms: int, to_ms: int) -> bytes:
    """segments: (WAV key, its start on the timeline in ms), in order and contiguous."""
    from_ms, to_ms = max(0, from_ms), max(from_ms + 1, to_ms)
    parts: list[np.ndarray] = []
    sr = 48000
    for key, seg_start in segments:
        path = cas.path(key)
        a = max(from_ms, seg_start)
        audio, sr = read_wav_range(path, int((a - seg_start) * sr / 1000),
                                   int((to_ms - a) * sr / 1000))  # fmt: skip
        if len(audio) and a < to_ms:
            # keep only the part of this segment that lies before the next one starts
            parts.append(audio)
    if not parts:
        return b""
    clip = np.concatenate(parts)[: int((to_ms - from_ms) * sr / 1000)]
    wav = cas.tmp_path(".wav")
    mp3 = cas.tmp_path(".mp3")
    try:
        write_wav(wav, clip, sr)
        run_ffmpeg(["-i", str(wav), "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)])
        return mp3.read_bytes()
    finally:
        wav.unlink(missing_ok=True)
        mp3.unlink(missing_ok=True)


def segments_for(tiles: list[dict], from_ms: int, to_ms: int) -> list[tuple[str, int]]:
    """The tiles (dicts with start_ms, end_ms, audio_key) that overlap [from_ms, to_ms)."""
    return [(t["audio_key"], t["start_ms"]) for t in tiles
            if t["start_ms"] < to_ms and t["end_ms"] > from_ms]  # fmt: skip
