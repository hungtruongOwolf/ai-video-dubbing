"""ffmpeg-based steps: decode audio, time-stretch, loudness normalisation.

ffmpeg is called with an argument list (no shell), so paths with spaces or non-ASCII
characters work.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from dub.contracts.v1 import AudioRes, DecodeReq, MediaInfo, StretchReq
from dub.ops.audio_io import duration_ms
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


def stretch(req: StretchReq, cas: CAS) -> AudioRes:
    """Change speed without changing pitch (ffmpeg atempo) and convert to 48 kHz mono.

    rate > 1 means faster (shorter). The pod version will use rubberband R3 for better quality.
    """
    src = cas.path(req.audio_key)
    out = cas.tmp_path(".wav")
    args = ["-i", str(src)]
    if abs(req.rate - 1.0) > 1e-3:
        args += ["-af", f"atempo={req.rate:.5f}"]
    args += ["-ac", "1", "-ar", str(req.sample_rate), "-c:a", "pcm_s16le", str(out)]
    run_ffmpeg(args)
    dur = duration_ms(out)
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
