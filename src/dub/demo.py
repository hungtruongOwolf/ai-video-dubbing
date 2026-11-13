"""`dub demo`: build a small test video you can dub right away.

It is a ~1 minute "kitchen show" interview between a host (S1) and an older guest (S2).
The picture is an ffmpeg test pattern; the speech is a synthetic buzz, one burst per word,
over a soft music pad. Next to the video we write `<name>.words.json`, the exact transcript
with word timings and speakers. The fake ASR replays it, so segmentation, the cast sheet,
translation (fake or real Claude) and timing all work on realistic data.

The script deliberately contains:
- a sentence longer than 15 s (must be split),
- a very short sentence ("Oh no!", must be merged),
- a pause longer than 1.5 s (mandatory boundary),
- forms-of-address evidence: "Mr. Tu", "sir"; and a name with no gender evidence ("Lan").
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from dub.contracts.v1 import Word
from dub.ops.audio_io import write_wav
from dub.ops.media import run_ffmpeg
from dub.ops.synth import pad, voice
from dub.ops.timing import count_units

SR = 48000

# (speaker, text, pause after the line in seconds)
SCRIPT: list[tuple[str, str, float]] = [
    ("S1", "Good evening everyone, and welcome back to our little kitchen show.", 0.5),
    (
        "S1",
        "Today I am sitting with a very special guest, Mr. Tu, who has been cooking for "
        "forty years.",
        0.8,
    ),
    ("S2", "Thank you, Lan. You make me sound so old!", 0.7),
    ("S1", "Forty years is a lot of practice, sir. What was the first dish you ever cooked?", 0.9),
    (
        "S2",
        "Rice porridge, for my mother, when she was sick. I was only nine years old, and I "
        "burned the bottom of the pot, so the whole house smelled of smoke for two days, "
        "but I was far too proud to start again from the beginning.",
        1.8,
    ),
    ("S2", "She ate every spoon anyway.", 0.7),
    ("S1", "Oh no! That is so sweet.", 0.6),
    ("S2", "She said it was the best porridge in the world, which was a kind lie.", 0.8),
    ("S1", "And what will you teach us today?", 0.7),
    (
        "S2",
        "A simple fish soup, the way my grandmother made it, with tamarind and a lot of patience.",
        1.0,
    ),
]

# What a voice age/gender model might say. Weak evidence, on purpose.
VOICES = [
    {"speaker": "S1", "gender": "female", "gender_conf": 0.91, "age_band": "adult"},
    {"speaker": "S2", "gender": "male", "gender_conf": 0.88, "age_band": "senior"},
]
PITCH = {"S1": 205.0, "S2": 118.0}


def build_words(script=SCRIPT, start_s: float = 0.6) -> list[Word]:
    words: list[Word] = []
    t = start_s
    for speaker, text, pause in script:
        for tok in text.split():
            syl = count_units(re.sub(r"[^\w]", "", tok) or tok, "syllable")
            dur = 0.12 + 0.17 * syl
            words.append(
                Word(
                    w=tok, start_ms=round(t * 1000), end_ms=round((t + dur) * 1000), speaker=speaker
                )
            )
            t += dur + 0.05
            if tok[-1] in ",;:":
                t += 0.18
            elif tok[-1] in ".!?":
                t += 0.3
        t += pause
    return words


def render_audio(words: list[Word], tail_s: float = 1.0) -> np.ndarray:
    total_ms = words[-1].end_ms + int(tail_s * 1000)
    mix = np.repeat(pad(total_ms)[:, None], 2, axis=1)
    for i, w in enumerate(words):
        burst = voice(w.end_ms - w.start_ms, PITCH.get(w.speaker or "S1", 160.0), seed=i)
        s = int(w.start_ms * SR / 1000)
        mix[s : s + len(burst)] += burst[:, None]
    return mix


def make_demo(out_dir: Path, name: str = "interview") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    words = build_words()
    audio = render_audio(words)
    wav = out_dir / f"{name}.tmp.wav"
    write_wav(wav, audio, SR)
    video = out_dir / f"{name}.mp4"
    dur = len(audio) / SR
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=25:duration={dur:.3f}",
            "-i",
            str(wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(video),
        ]
    )
    wav.unlink(missing_ok=True)
    sidecar = out_dir / f"{name}.words.json"
    sidecar.write_text(
        json.dumps(
            {"lang": "en", "words": [w.model_dump() for w in words], "voices": VOICES},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return video, sidecar
