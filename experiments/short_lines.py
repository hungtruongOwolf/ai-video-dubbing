"""How much time does OmniVoice need for a short line? (M8)

Very short dub lines (0.4-1.3 s) came out garbled on a zh -> vi drama. Our length plan is
linear (syllables / units_per_s); OmniVoice's own estimator gives short texts much more time
(below ~2 s it returns 50 * (frames / 50) ** (1/3) frames). Here short Vietnamese lines of 1-6
syllables are voiced at several multiples of the linear plan, with the reference voices of
the E1 set, and back-checked (PhoWhisper): CER per multiple and length.

Pod (workers running):  envs/core/bin/python experiments/short_lines.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, run_worker, save_json  # noqa: E402

LINES = [
    "Gầy rồi.", "Đốt rồi sao?", "Tạ lão gia!", "Đợi mẹ về.", "Thưởng!", "Vâng ạ.",
    "Cực không em?", "A Nguyên.", "Đi đi, nhanh lên.", "Ừ, con ở một mình.",
    "Không được đâu.", "Hay quá.", "Mau lên đi.", "Thế thì tốt rồi.", "Ngươi tìm ai?",
    "Biết rồi.", "Được rồi.", "Cảm ơn anh.", "Con ngủ đi.", "Ai đó?",
]  # fmt: skip
MULTIPLES = [1.0, 1.3, 1.6, 2.0, 2.5]
UNITS_PER_S = 4.7


def _ws():
    from dub.store import Workspace

    return Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                          db_path=Path("/tmp/exp.sqlite"))  # fmt: skip


def main() -> None:
    from dub.contracts.v1 import AudioReq, TrimReq
    from dub.ops import media
    from dub.ops.align import cer
    from dub.ops.timing import count_units

    cas = _ws().cas
    voices = []
    for f in sorted(data_dir().glob("e1_*.*")):
        if f.suffix not in (".wav", ".flac"):
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
    for m in MULTIPLES:
        errs, rows = [], []
        for vi, (vname, key, vtext) in enumerate(voices):
            for li, line in enumerate(LINES):
                syl = count_units(line, "syllable")
                target = round(syl / UNITS_PER_S * 1000 * m)
                out = run_worker("tts", "omnivoice", "tts.synthesize", {
                    "unit_id": f"short-{vi}-{li}", "text": line, "lang": "vi", "speaker": vname,
                    "voice_key": key, "voice_text": vtext, "target_ms": target,
                })  # fmt: skip
                tr = media.trim(TrimReq(audio_key=out["audio_key"]), cas)
                k16 = media.to_mono16k(AudioReq(audio_key=tr.audio_key), cas).audio_key
                heard = run_worker("asr", "phowhisper_large_ct2", "asr.backcheck",
                                   {"audio_key": k16, "lang": "vi"})  # fmt: skip
                e = cer(line, " ".join(w["w"] for w in heard["words"]))
                errs.append(e)
                rows.append({"line": line, "syl": syl, "target_ms": target, "cer": round(e, 3)})
        by_syl = {}
        for r in rows:
            by_syl.setdefault(r["syl"], []).append(r["cer"])
        res[str(m)] = {
            "cer_mean": round(float(np.mean(errs)), 3),
            "ok_share": round(float(np.mean([e <= 0.15 for e in errs])), 3),
            "cer_by_syllables": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_syl.items())},
        }
        print(m, json.dumps(res[str(m)]), flush=True)
        save_json(data_dir() / "results" / "short_lines.json", res)


if __name__ == "__main__":
    main()
