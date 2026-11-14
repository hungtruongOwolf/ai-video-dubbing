"""Is source-language speech left in the background bed? (M8)

The dub is mixed over the separated instrumental. Whatever the separator left of the original
voice there is heard under the dub as "the original language is still audible". This script
transcribes the bed alone in the source language: words Whisper recognises there with some
confidence are leaked speech, with their times.

Pod (workers running):  envs/core/bin/python experiments/leak_check.py SRC_KEY --lang zh
                        [--param mel_roformer.KEY=VALUE ...] (separation variants)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import run_worker  # noqa: E402

MIN_PROB = 0.5  # a word heard this surely is speech, not music


def _ws():
    from dub.store import Workspace

    return Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                          db_path=Path("/tmp/exp.sqlite"))  # fmt: skip


def leaked_words(bed16_key: str, lang: str) -> list[dict]:
    tl = run_worker(
        "asr", "whisper_large_v3", "asr.transcribe", {"audio_key": bed16_key, "lang": lang}
    )
    return [w for w in tl["words"] if (w.get("conf") or 0) >= MIN_PROB]


def regions(words: list[dict], gap_ms: int = 1500) -> list[tuple[int, int, str]]:
    out: list[list] = []
    for w in words:
        if out and w["start_ms"] - out[-1][1] <= gap_ms:
            out[-1][1] = w["end_ms"]
            out[-1][2] += w["w"]
        else:
            out.append([w["start_ms"], w["end_ms"], w["w"]])
    return [(a, b, t) for a, b, t in out]


def main() -> None:
    from dub.contracts.v1 import AudioReq, DecodeReq
    from dub.ops import media

    ap = argparse.ArgumentParser()
    ap.add_argument("src_key", help="CAS key of the uploaded audio track (.m4a/.mka)")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--param", action="append", default=[], help="mel_roformer.KEY=VALUE")
    a = ap.parse_args()
    cas = _ws().cas
    info = media.decode(DecodeReq(src_key=a.src_key), cas)
    params = {}
    for p in a.param:
        k, _, v = p.partition("=")
        params[k.split(".", 1)[1]] = json.loads(v) if v[:1].isdigit() else v
    st = run_worker("sep", "mel_roformer", "audio.separate", {"audio_key": info.a48k_key},
                    params=params)  # fmt: skip
    res = {}
    for name in ("instrumental_key", "vocals_key"):
        k16 = media.to_mono16k(AudioReq(audio_key=st[name]), cas).audio_key
        words = leaked_words(k16, a.lang)
        res[name] = {"words": len(words), "regions": regions(words)}
    bed = res["instrumental_key"]
    print(f"bed: {bed['words']} confident {a.lang} words in {len(bed['regions'])} regions "
          f"(vocals stem, for scale: {res['vocals_key']['words']} words)")  # fmt: skip
    for s, e, t in bed["regions"]:
        print(f"  {s / 1000:7.1f}-{e / 1000:7.1f}s  {t[:60]}")
    print(json.dumps({"bed_key": st["instrumental_key"], "vocals_key": st["vocals_key"]}))


if __name__ == "__main__":
    main()
