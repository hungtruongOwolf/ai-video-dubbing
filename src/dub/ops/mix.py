"""Mixing in fixed 30 s tiles, so an edit only remixes the tiles it touches.

    mix_tile  bed slice + every clip overlapping the tile   (cached per tile)
    concat    tiles -> one WAV                             (export only)
    master    loudness normalisation + AAC                 (export only)

Tile boundaries are a fixed grid, independent of where clips land, so moving one line never
shifts the other tiles. A clip that crosses a boundary is rendered in both tiles, each taking
its own part. Everything here is sample-wise (fades, soft clipper), so the tiles join without
seams.
"""

from __future__ import annotations

import wave

import numpy as np

from dub.contracts.v1 import AudioRes, ConcatReq, MasterReq, Tile, TileReq
from dub.ops.audio_io import duration_ms, read_wav, read_wav_range, write_wav
from dub.ops.media import probe, run_ffmpeg
from dub.store.cas import CAS

SR = 48000
FADE_MS = 8  # short fade in/out so line starts and ends do not click
TILE_MS = 30_000
KNEE = 0.8  # soft clipper: untouched below, smoothly limited above


def plan_tiles(duration_ms: int, tile_ms: int = TILE_MS) -> list[tuple[int, int]]:
    return [(t, min(t + tile_ms, duration_ms)) for t in range(0, max(duration_ms, 1), tile_ms)]


def _fade(clip: np.ndarray) -> np.ndarray:
    n = min(len(clip) // 2, int(SR * FADE_MS / 1000))
    if n <= 0:
        return clip
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    clip = clip.copy()
    clip[:n] *= ramp
    clip[-n:] *= ramp[::-1]
    return clip


def soft_clip(x: np.ndarray) -> np.ndarray:
    """Sample-wise limiter: identity below KNEE, tanh-shaped above, never beyond 1.0."""
    a = np.abs(x)
    over = a > KNEE
    if over.any():
        x = x.copy()
        x[over] = np.sign(x[over]) * (KNEE + (1 - KNEE) * np.tanh((a[over] - KNEE) / (1 - KNEE)))
    return x


def mix_tile(req: TileReq, cas: CAS) -> Tile:
    t0 = int(req.start_ms * SR / 1000)
    t1 = int(req.end_ms * SR / 1000)
    bed, sr = read_wav_range(cas.path(req.bed_key), t0, t1 - t0)
    if sr != SR:
        raise ValueError(f"The bed must be 48 kHz, got {sr}")
    out = np.zeros((t1 - t0, 2), dtype=np.float32)
    n = min(len(bed), t1 - t0)
    out[:n] = bed[:n, :2] if bed.shape[1] >= 2 else np.repeat(bed[:n], 2, axis=1)
    out *= 10 ** (req.bed_gain_db / 20)

    for p in req.placements:
        if not p.audio_key:
            continue
        clip, csr = read_wav(cas.path(p.audio_key))
        if csr != SR:
            raise ValueError(f"Clip {p.unit_id} must be 48 kHz, got {csr}")
        mono = _fade(clip.mean(axis=1))
        if p.gain_db:
            mono = mono * np.float32(10 ** (p.gain_db / 20))
        c0 = int(p.start_ms * SR / 1000)  # clip start on the timeline
        a, b = max(c0, t0), min(c0 + len(mono), t1)  # overlap with the tile
        if b > a:
            out[a - t0 : b - t0] += mono[a - c0 : b - c0, None]

    tmp = cas.tmp_path(".wav")
    write_wav(tmp, soft_clip(out), SR)
    return Tile(start_ms=req.start_ms, end_ms=req.end_ms, audio_key=cas.put_file(tmp, move=True))


def concat(req: ConcatReq, cas: CAS) -> AudioRes:
    """Join WAV tiles (same format) frame by frame, without decoding into one big array."""
    out = cas.tmp_path(".wav")
    with wave.open(str(out), "wb") as w:
        for i, key in enumerate(req.keys):
            with wave.open(str(cas.path(key)), "rb") as r:
                if i == 0:
                    w.setparams(r.getparams())
                while frames := r.readframes(SR * 10):
                    w.writeframes(frames)
    dur = duration_ms(out)
    return AudioRes(audio_key=cas.put_file(out, move=True), dur_ms=dur)


def integrated_lufs(path) -> float | None:
    """EBU R128 integrated loudness of a file (ffmpeg's ebur128 filter, ~3 s per hour)."""
    import re
    import subprocess

    from dub.ops.media import _bin

    p = subprocess.run(
        [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-i", str(path),
         "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    m = re.findall(r"I:\s+(-?[\d.]+|-inf) LUFS", p.stderr)
    if not m or m[-1] == "-inf":
        return None
    return float(m[-1])


def master(req: MasterReq, cas: CAS) -> AudioRes:
    """Loudness-normalise the whole track (EBU R128) and encode AAC in an .m4a.

    One static gain to the target integrated loudness, then a true-peak limiter at -1.5 dBTP:
    the mix keeps its dynamics (ffmpeg's dynamic loudnorm rode the gain and, at 192 kHz
    internally, took ~50 s per hour of audio). The AAC "fast" coder halves the encode time
    (81 -> 44 s per hour on the pod) at a bitrate where the difference is not audible.
    """
    src = cas.path(req.audio_key)
    lufs = integrated_lufs(src)
    gain = 0.0 if lufs is None else max(-30.0, min(30.0, req.loudness_lufs - lufs))
    limit = 10 ** (-1.5 / 20)
    out = cas.tmp_path(".m4a")
    run_ffmpeg([
        "-i", str(src),
        "-af", f"volume={gain:.2f}dB,alimiter=limit={limit:.4f}:level=disabled:latency=1",
        "-ar", "48000", "-c:a", "aac", "-aac_coder", "fast", "-b:a", req.bitrate,
        "-movflags", "+faststart", str(out),
    ])  # fmt: skip
    dur = round(float(probe(out).get("format", {}).get("duration", 0)) * 1000)
    return AudioRes(audio_key=cas.put_file(out, move=True), dur_ms=dur)
