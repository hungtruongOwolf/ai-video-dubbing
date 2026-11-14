"""Read subtitles burned into the picture (laptop side; the video never leaves the laptop).

1. Find the subtitle band: full OCR (detection + recognition) on 12 frames spread over the
   video, bottom 40% of the picture. Text in the same band on at least 40% of them means
   burned-in subtitles; the band is the union of those boxes plus a margin.
2. Crop the band at 4 frames per second (ffmpeg) and cut it into events where the picture
   of the band changes (mean absolute difference of a 200 x 20 grey thumbnail > 6).
3. Recognition only (no detection) on up to 3 frames of each event; the surest reading wins.
   Consecutive events with the same text become one cue.

Speed on a laptop CPU: ~10 s for a 5-minute video (full OCR of every frame took 964 s).
Needs the `ocr` dependency group (RapidOCR, ONNX Runtime; Apache-2.0, PaddleOCR models). Its
default model reads Chinese and Latin letters, so it is used for Chinese sources.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from dub.contracts.v1 import HardsubCue
from dub.ops.media import probe, run_ffmpeg

FPS = 4.0
SAMPLES = 12
BAND_TOP = 0.6  # look for subtitles in the bottom 40% of the picture
MIN_SHARE = 0.4  # of the sampled frames must show text in the band
CHANGE = 6.0  # grey-level difference of the band thumbnail that starts a new event
MIN_SCORE = 0.6
_KEEP = re.compile(r"[\w　-〿＀-￯，。！？、；：“”‘’（）]")


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _video_size(video: Path) -> tuple[int, int, float] | None:
    info = probe(video)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        return None
    return int(v["width"]), int(v["height"]), float(info.get("format", {}).get("duration", 0))


def find_band(video: Path, ocr=None) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) of the burned-in subtitle band, or None."""
    size = _video_size(video)
    if not size or size[2] <= 0:
        return None
    width, height, dur = size
    ocr = ocr or _engine()
    top0 = int(height * BAND_TOP)
    boxes = []
    hits = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(SAMPLES):
            t = dur * (i + 0.5) / SAMPLES
            img = Path(tmp) / f"s{i}.png"
            run_ffmpeg(["-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"crop={width}:{height - top0}:0:{top0}", str(img)])  # fmt: skip
            res, _ = ocr(str(img))
            found = False
            for box, text, score in res or []:
                if float(score) >= 0.8 and len(_clean(text)) >= 2:
                    xs, ys = [p[0] for p in box], [p[1] for p in box]
                    boxes.append((min(xs), min(ys) + top0, max(xs), max(ys) + top0))
                    found = True
            hits += found
    if hits < MIN_SHARE * SAMPLES or not boxes:
        return None
    # Subtitles sit at the same height in every frame; text in the scene (signs, a logo)
    # does not. Keep the boxes around the median line.
    mid = float(np.median([(b[1] + b[3]) / 2 for b in boxes]))
    tall = float(np.median([b[3] - b[1] for b in boxes]))
    line = [b for b in boxes if abs((b[1] + b[3]) / 2 - mid) <= 0.75 * tall]
    if len(line) < MIN_SHARE * SAMPLES:
        return None
    y0 = int(max(0, min(b[1] for b in line) - 10))
    y1 = int(min(height, max(b[3] for b in line) + 10))
    # a longer line than the sampled ones must fit: at least the middle 80% of the picture
    x0 = int(max(0, min(min(b[0] for b in line) - 60, width * 0.1)))
    x1 = int(min(width, max(max(b[2] for b in line) + 60, width * 0.9)))
    return x0, y0, max(16, x1 - x0) // 2 * 2, max(16, y1 - y0) // 2 * 2


def _clean(text: str) -> str:
    return "".join(_KEEP.findall(text.replace("#", "")))


def read_cues(video: Path, band: tuple[int, int, int, int], ocr=None) -> list[HardsubCue]:
    from PIL import Image

    ocr = ocr or _engine()
    x, y, w, h = band
    tmp = Path(tempfile.mkdtemp(prefix="dub-hardsubs-"))
    try:
        run_ffmpeg(["-i", str(video), "-vf", f"fps={FPS},crop={w}:{h}:{x}:{y}", "-q:v", "3",
                    str(tmp / "f%06d.jpg")])  # fmt: skip
        files = sorted(tmp.glob("f*.jpg"))
        thumbs = [np.asarray(Image.open(f).convert("L").resize((200, 20)), dtype=np.float32)
                  for f in files]  # fmt: skip
        events, start = [], 0
        for i in range(1, len(files) + 1):
            if i == len(files) or np.abs(thumbs[i] - thumbs[i - 1]).mean() > CHANGE:
                events.append((start, i))
                start = i
        cues: list[HardsubCue] = []
        for a, b in events:
            picks = sorted({a + (b - a) // 4, (a + b - 1) // 2, b - 1 - (b - a) // 4})
            best = ("", 0.0)
            for k in picks:
                res, _ = ocr(str(files[k]), use_det=False, use_cls=False, use_rec=True)
                if res and float(res[0][1]) > best[1]:
                    best = (_clean(res[0][0]), float(res[0][1]))
            text, score = best
            if score < MIN_SCORE or not any(ch.isalnum() for ch in text):
                continue
            s_ms, e_ms = round(a * 1000 / FPS), round(b * 1000 / FPS)
            if cues and cues[-1].text == text and s_ms - cues[-1].end_ms <= 300:
                cues[-1] = cues[-1].model_copy(
                    update={"end_ms": e_ms, "conf": max(cues[-1].conf, round(score, 3))}
                )
            else:
                cues.append(HardsubCue(start_ms=s_ms, end_ms=e_ms, text=text, conf=round(score, 3)))
        return cues
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def burned_subtitles(video: Path, log=print) -> list[HardsubCue]:
    """The burned-in subtitles of a video, or [] (no band found, or OCR not installed)."""
    if not available():
        log("[laptop] burned-in subtitles not read: install the `ocr` dependency group")
        return []
    ocr = _engine()
    band = find_band(video, ocr)
    if band is None:
        return []
    cues = read_cues(video, band, ocr)
    log(f"[laptop] burned-in subtitles: {len(cues)} cues read in the band {band}")
    return cues
