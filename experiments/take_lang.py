"""Does any dubbed line speak the source language? (M8)

Whisper (language auto) hears every placed clip of a finished job. A line whose clip it hears
as the source language, or whose transcript holds characters of the source script, leaks the
original language into the dub (a voice clone continuing its reference, untranslated text).
Lines kept original are skipped.

Pod (workers running):  envs/core/bin/python experiments/take_lang.py JOB_DIR [--src zh]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import run_worker  # noqa: E402

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")


def main() -> None:
    from dub.contracts.v1 import AudioReq
    from dub.ops import media
    from dub.store import Workspace

    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--src", default="zh")
    a = ap.parse_args()
    cas = Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                         db_path=Path("/tmp/exp.sqlite")).cas  # fmt: skip
    rows = json.loads((a.job_dir / "units.json").read_text(encoding="utf-8"))
    leaks, checked = [], 0
    for r in rows:
        p = r.get("placement") or {}
        if not p.get("audio_key") or p.get("variant") == "original":
            continue
        if not cas.exists(p["audio_key"]):
            continue
        k16 = media.to_mono16k(AudioReq(audio_key=p["audio_key"]), cas).audio_key
        tl = run_worker(
            "asr", "whisper_large_v3", "asr.transcribe", {"audio_key": k16, "lang": "auto"}
        )
        text = "".join(w["w"] for w in tl["words"])
        checked += 1
        if tl.get("lang") == a.src or CJK.search(text):
            leaks.append(
                {
                    "unit": p["unit_id"],
                    "start_s": p["start_ms"] / 1000,
                    "heard_lang": tl.get("lang"),
                    "heard": text[:60],
                    "text": p["text"][:60],
                }
            )
    print(
        f"{checked} dubbed lines checked, {len(leaks)} heard as {a.src} or with {a.src} characters"
    )
    for x in leaks:
        head = f"  {x['start_s']:7.1f}s {x['unit']:<16} heard {x['heard_lang']}"
        print(f"{head}: {x['heard']}  |  {x['text']}")


if __name__ == "__main__":
    main()
