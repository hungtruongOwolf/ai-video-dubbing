"""ffmpeg-based steps: decode audio, time-stretch, loudness normalisation.

ffmpeg is called with an argument list (no shell), so paths with spaces or non-ASCII
characters work.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from dub.contracts.v1 import (
    AudioReq,
    AudioRes,
    CutReq,
    DecodeReq,
    EncodeReq,
    MediaInfo,
    StretchReq,
    TrimReq,
    TrimRes,
)
from dub.ops.audio_io import duration_ms, read_wav, resample_int, write_wav
from dub.store.cas import CAS


class FFmpegError(RuntimeError):
    pass


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(
            f"'{name}' not found. Install ffmpeg and reopen the terminal "
            "(Windows: winget install Gyan.FFmpeg; Ubuntu: sudo apt install ffmpeg)."
        )
    return path


def run_ffmpeg(args: list[str]) -> None:
    cmd = [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise FFmpegError(f"ffmpeg failed (exit {p.returncode}): {p.stderr.strip()[-2000:]}")


def probe(path: Path) -> dict:
    cmd = [
        _bin("ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise FFmpegError(f"ffprobe cannot read {path}: {p.stderr.strip()}")
    return json.loads(p.stdout)


# ---------------------------------------------------------------- local ops (memoized)


def decode(req: DecodeReq, cas: CAS) -> MediaInfo:
    """Original audio -> 48 kHz stereo WAV (for mixing) + 16 kHz mono WAV (for ASR)."""
    src = cas.path(req.src_key)
    a48 = cas.tmp_path(".wav")
    a16 = cas.tmp_path(".wav")
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-map",
            "0:a:0",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(a48),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(a16),
        ]
    )
    dur = duration_ms(a48)
    return MediaInfo(
        src_key=req.src_key,
        duration_ms=dur,
        a48k_key=cas.put_file(a48, move=True),
        a16k_key=cas.put_file(a16, move=True),
    )


@functools.cache
def stretch_engine() -> str:
    """'rubberband' when this ffmpeg has the librubberband filter (better quality), else
    'atempo'. The pipeline puts the engine into the request, so it is part of the cache key."""
    try:
        p = subprocess.run(
            [_bin("ffmpeg"), "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FFmpegError:
        return "atempo"
    return "rubberband" if " rubberband " in p.stdout else "atempo"


def stretch(req: StretchReq, cas: CAS) -> AudioRes:
    """Change speed without changing pitch, and convert to 48 kHz mono.

    rate > 1 means faster (shorter).
    """
    src = cas.path(req.audio_key)
    out = cas.tmp_path(".wav")
    if abs(req.rate - 1.0) <= 1e-3 and src.suffix == ".wav":
        audio, sr = read_wav(src)
        ratio = _integer_ratio(sr, req.sample_rate)
        if ratio:  # no tempo change: only a resample, done in numpy
            write_wav(out, resample_int(audio.mean(axis=1), *ratio), req.sample_rate)
            dur = duration_ms(out)
            return AudioRes(audio_key=cas.put_file(out, move=True), dur_ms=dur)
    args = ["-i", str(src)]
    if abs(req.rate - 1.0) > 1e-3:
        if req.engine == "rubberband":
            args += ["-af", f"rubberband=tempo={req.rate:.5f}:pitchq=quality:transients=smooth"]
        else:
            args += ["-af", f"atempo={req.rate:.5f}"]
    args += ["-ac", "1", "-ar", str(req.sample_rate), "-c:a", "pcm_s16le", str(out)]
    run_ffmpeg(args)
    dur = duration_ms(out)
    return AudioRes(audio_key=cas.put_file(out, move=True), dur_ms=dur)


def _integer_ratio(src: int, dst: int) -> tuple[int, int] | None:
    for up in (1, 2):
        for down in (1, 2, 3, 6):
            if src * up == dst * down:
                return up, down
    return None


def cut(req: CutReq, cas: CAS) -> AudioRes:
    """A slice of a 16-bit WAV, same format (a line's original vocals, mixed as they are)."""
    from dub.ops.audio_io import read_wav_range

    src = cas.path(req.audio_key)
    with wave.open(str(src), "rb") as w:
        sr = w.getframerate()
    a = int(req.start_ms * sr / 1000)
    audio, sr = read_wav_range(src, a, max(0, int(req.end_ms * sr / 1000) - a))
    out = cas.tmp_path(".wav")
    write_wav(out, audio, sr)
    dur = duration_ms(out)
    return AudioRes(audio_key=cas.put_file(out, move=True), dur_ms=dur)


def to_mono16k(req: AudioReq, cas: CAS) -> AudioRes:
    """Any WAV -> 16 kHz mono 16-bit WAV (what ASR and the VAD read)."""
    out = cas.tmp_path(".wav")
    path = cas.path(req.audio_key)
    if path.suffix == ".wav" and path.stat().st_size < 50_000_000:  # clips: numpy is faster
        audio, sr = read_wav(path)
        ratio = _integer_ratio(sr, 16000)
        if ratio:
            write_wav(out, resample_int(audio.mean(axis=1), *ratio), 16000)
            dur = duration_ms(out)
            return AudioRes(audio_key=cas.put_file(out, move=True), dur_ms=dur)
    run_ffmpeg(
        [
            "-i",
            str(cas.path(req.audio_key)),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    )
    dur = duration_ms(out)
    return AudioRes(audio_key=cas.put_file(out, move=True), dur_ms=dur)


TRIM_FRAME_MS = 10
TRIM_REL_DB = 40.0  # a frame is "voice" if within 40 dB of the loudest frame...
TRIM_FLOOR_DB = -55.0  # ...and louder than this


def voice_edges(mono: np.ndarray, sr: int) -> tuple[int, int] | None:
    """(first, last) sample of the voice in a clean clip, or None if it is silent."""
    hop = max(1, sr * TRIM_FRAME_MS // 1000)
    n = len(mono) // hop
    if n == 0:
        return None
    frames = mono[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) + 1e-9)
    thr = max(TRIM_FLOOR_DB, db.max() - TRIM_REL_DB)
    on = np.flatnonzero(db > thr)
    if on.size == 0:
        return None
    return int(on[0] * hop), int(min(len(mono), (on[-1] + 1) * hop))


def trim(req: TrimReq, cas: CAS) -> TrimRes:
    """Cut leading and trailing silence from a TTS clip, so the voice starts exactly where
    the clip is placed (onset sync) and the fit works with the length of the voice itself."""
    audio, sr = read_wav(cas.path(req.audio_key))
    mono = audio.mean(axis=1)
    edges = voice_edges(mono, sr)
    total = round(len(mono) * 1000 / sr)
    if edges is None:
        return TrimRes(audio_key=req.audio_key, dur_ms=total, lead_ms=0, tail_ms=0)
    pad = sr * req.pad_ms // 1000
    a, b = max(0, edges[0] - pad), min(len(mono), edges[1] + pad)
    if a == 0 and b == len(mono):
        return TrimRes(audio_key=req.audio_key, dur_ms=total, lead_ms=0, tail_ms=0)
    out = cas.tmp_path(".wav")
    write_wav(out, audio[a:b], sr)
    return TrimRes(
        audio_key=cas.put_file(out, move=True),
        dur_ms=round((b - a) * 1000 / sr),
        lead_ms=round(a * 1000 / sr),
        tail_ms=round((len(mono) - b) * 1000 / sr),
    )


def encode(req: EncodeReq, cas: CAS) -> AudioRes:
    """WAV -> AAC in an .m4a container: ~85 MB per hour instead of ~660 MB, so the result
    downloads quickly; the laptop then muxes it with stream copy."""
    out = cas.tmp_path(".m4a")
    run_ffmpeg(
        [
            "-i",
            str(cas.path(req.audio_key)),
            "-c:a",
            req.codec,
            "-b:a",
            req.bitrate,
            "-movflags",
            "+faststart",
            str(out),
        ]
    )
    info = probe(out)
    dur = round(float(info.get("format", {}).get("duration", 0)) * 1000)
    return AudioRes(audio_key=cas.put_file(out, move=True), dur_ms=dur)


def loudnorm(src: Path, dst: Path, lufs: float) -> None:
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-af",
            f"loudnorm=I={lufs}:TP=-1.5:LRA=11",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
