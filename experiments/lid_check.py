"""Accent check: is Whisper's language detection alone as good as a full transcription?

The QA gate asks Whisper (language auto) what language each dubbed take sounds like; a take
heard as the source language carries the actor's accent (M8). It used a full transcription
(beam 5, word times) to read one field, the language. `detect_language` runs the same
detection (same VAD, the first 30 s window) without decoding. Both are run on the dubbed
takes of a finished job: do they agree, and how long does each take?

Pod (asr env):
    envs/asr/bin/python experiments/lid_check.py ws/jobs/<job>/units.json --cas ws/cas
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def pcm16k(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def main() -> None:
    from faster_whisper import WhisperModel

    ap = argparse.ArgumentParser()
    ap.add_argument("units", type=Path)
    ap.add_argument("--cas", type=Path, default=Path("/workspace/dub/ws/cas"))
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--min-ms", type=int, default=1500)
    a = ap.parse_args()
    rows = json.loads(a.units.read_text(encoding="utf-8"))
    takes = [r["placement"] for r in rows if r["placement"]["variant"] != "original"]
    takes = [p for p in takes if p["end_ms"] - p["start_ms"] >= a.min_ms]
    model = WhisperModel(a.model, device="cuda", compute_type="float16")
    model.detect_language(audio=np.zeros(16000, dtype=np.float32))  # warm-up
    agree = 0
    t_full = t_det = 0.0
    out = []
    for p in takes:
        key = p["audio_key"]
        audio = pcm16k(a.cas / key[:2] / key)
        t0 = time.perf_counter()
        segs, info = model.transcribe(audio, language=None, word_timestamps=True, vad_filter=True,
                                      beam_size=5, condition_on_previous_text=False)  # fmt: skip
        words = [w for s in segs for w in (s.words or [])]
        t1 = time.perf_counter()
        lang, prob, _ = model.detect_language(audio=audio, vad_filter=True)
        t2 = time.perf_counter()
        full = info.language if words else None
        agree += full is None or full == lang
        t_full += t1 - t0
        t_det += t2 - t1
        out.append({"unit": p["unit_id"], "full": full, "detect": lang, "prob": round(prob, 3)})
    n = max(1, len(takes))
    print(json.dumps({
        "takes": len(takes), "agree": agree,
        "full_ms": round(1000 * t_full / n), "detect_ms": round(1000 * t_det / n),
        "full_langs": {k: sum(o["full"] == k for o in out) for k in {o["full"] for o in out}},
        "detect_langs": {k: sum(o["detect"] == k for o in out) for k in {o["detect"] for o in out}},
        "differ": [o for o in out if o["full"] is not None and o["full"] != o["detect"]],
    }, default=str))  # fmt: skip


if __name__ == "__main__":
    main()
