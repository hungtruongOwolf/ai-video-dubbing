"""How much time does OmniVoice need for a short line? (M8)

Very short dub lines (0.4-1.3 s) came out garbled on a zh -> vi drama. Our length plan is
linear (syllables / units_per_s); OmniVoice's own estimator gives short texts much more time
(below ~2 s it returns 50 * (frames / 50) ** (1/3) frames). Here short Vietnamese lines of 1-6
syllables are voiced at several multiples of the linear plan, with the reference voices of
the E1 set (English, Chinese) and, with `--refs`, the voices of a finished job (a zh drama),
and back-checked (PhoWhisper; text a clip that short cannot hold is dropped first,
`dub.ops.qa.plausible`): CER per multiple and length. Every take's transcript is saved.

Pod (workers running):  envs/core/bin/python experiments/short_lines.py [--refs JOB/refs.json]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dub.contracts.v1 import Word  # noqa: E402
from experiments.common import data_dir, run_worker, save_json  # noqa: E402

LINES = [
    "Trai.", "Thưởng!", "Gầy rồi.", "Vâng ạ.", "Đốt rồi sao?", "Đợi mẹ về.",
    "Ngươi tìm ai?", "Tám lạng.", "Đi đi, nhanh lên.", "Thế thì tốt rồi.",
    "Ừ, con ở một mình.", "Không có ai nói cả đâu.",
]  # fmt: skip
E1_VOICES = ("e1_main_f", "e1_zh_voice")
MULTIPLES = [1.0, 1.3, 1.6, 2.0, 2.5]
UNITS_PER_S = 4.7


def _ws():
    from dub.store import Workspace

    return Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                          db_path=Path("/tmp/exp.sqlite"))  # fmt: skip


def main() -> None:
    import argparse

    from dub.contracts.v1 import AudioReq, TrimReq
    from dub.ops import media
    from dub.ops.align import cer
    from dub.ops.qa import plausible
    from dub.ops.timing import count_units

    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", type=Path, help="refs.json of a finished job: its voices too")
    ap.add_argument("--multiples", default=",".join(map(str, MULTIPLES)))
    ap.add_argument(
        "--engine",
        default="omnivoice",
        choices=["omnivoice", "vieneu"],
        help="vieneu: VieNeu-TTS at its own pace (no target length; one pass)",
    )
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cas = _ws().cas
    voices = []
    if a.refs:
        for r in json.loads(a.refs.read_text(encoding="utf-8"))["refs"]:
            voices.append((f"job_{r['speaker']}", r["audio_key"], r["text"]))
    for f in sorted(data_dir().glob("e1_*.*")):
        if f.suffix not in (".wav", ".flac") or f.stem not in E1_VOICES:
            continue
        tmp = cas.tmp_path(".wav")
        media.run_ffmpeg(["-i", str(f), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(tmp)])
        key = cas.put_file(tmp, move=True)
        k16 = media.to_mono16k(AudioReq(audio_key=key), cas).audio_key
        text = " ".join(w["w"] for w in run_worker(
            "asr", "whisper_large_v3", "asr.transcribe", {"audio_key": k16, "lang": "auto"}
        )["words"])  # fmt: skip
        voices.append((f.stem, key, text))
    res: dict = {}
    multiples = [1.0] if a.engine == "vieneu" else [float(x) for x in a.multiples.split(",")]
    for m in multiples:
        errs, rows = [], []
        for vi, (vname, key, vtext) in enumerate(voices):
            for li, line in enumerate(LINES):
                syl = count_units(line, "syllable")
                target = round(syl / UNITS_PER_S * 1000 * m)
                req = {"unit_id": f"short-{vi}-{li}", "text": line, "lang": "vi",
                       "speaker": vname, "voice_key": key, "voice_text": vtext}  # fmt: skip
                if a.engine == "vieneu":
                    out = run_worker("tts2", "vieneu_v3_turbo", "tts.synthesize", req)
                else:
                    out = run_worker("tts", "omnivoice", "tts.synthesize",
                                     {**req, "target_ms": target})  # fmt: skip
                tr = media.trim(TrimReq(audio_key=out["audio_key"]), cas)
                k16 = media.to_mono16k(AudioReq(audio_key=tr.audio_key), cas).audio_key
                heard = run_worker("asr", "phowhisper_large_ct2", "asr.backcheck",
                                   {"audio_key": k16, "lang": "vi"})  # fmt: skip
                words = [Word.model_validate(w) for w in heard["words"]]
                kept = plausible(words, tr.dur_ms, line, "syllable", " ")
                e = min(1.0, cer(line, " ".join(w.w for w in kept)))
                errs.append(e)
                rows.append({"line": line, "syl": syl, "voice": vname, "target_ms": target,
                             "dur_ms": tr.dur_ms, "cer": round(e, 3),
                             "heard": " ".join(w.w for w in words)})  # fmt: skip
        by_syl = {}
        for r in rows:
            by_syl.setdefault(r["syl"], []).append(r["cer"])
        ok_by: dict = {}
        for r in rows:
            ok_by.setdefault(r["syl"], []).append(r["cer"] <= 0.15)
        res[str(m)] = {
            "cer_mean": round(float(np.mean(errs)), 3),
            "ok_share": round(float(np.mean([e <= 0.15 for e in errs])), 3),
            "cer_by_syllables": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_syl.items())},
            "ok_by_syllables": {k: round(float(np.mean(v)), 2) for k, v in sorted(ok_by.items())},
            "rows": rows,
        }
        print(m, json.dumps({k: v for k, v in res[str(m)].items() if k != "rows"}), flush=True)
        name = f"short_lines{'_' + a.engine if a.engine != 'omnivoice' else ''}{a.tag}.json"
        save_json(data_dir() / "results" / name, res)


if __name__ == "__main__":
    main()
