"""How foreign does the cloned voice sound? (M8)

On a zh -> vi drama, Whisper (language auto) heard 4 of 35 dubbed lines as Chinese although
PhoWhisper understood them (CER 0.03-0.08): Vietnamese spoken with the Chinese actor's
accent, the price of cloning a voice across languages. Variants, on the lines of a finished
job (its refs.json and units.json):
- omnivoice: as the job did (seed 0);
- omnivoice_g3: classifier-free guidance 3.0 instead of 2.0 (stronger text conditioning);
- omnivoice_best3: seeds 0, 1, 2, keeping the take Whisper hears as Vietnamese if any;
- vieneu: VieNeu-TTS v3 Turbo (a Vietnamese model cloning the same reference).
Reported per variant: share of lines heard as the source language, back-check CER
(PhoWhisper), and similarity to the reference clip (ECAPA).

Pod (workers running):  envs/core/bin/python experiments/accent.py JOB_DIR [--src zh]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, run_worker, save_json  # noqa: E402

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")


def main() -> None:
    from dub.contracts.v1 import AudioReq, TrimReq, Word
    from dub.ops import media
    from dub.ops.align import cer
    from dub.ops.qa import plausible
    from dub.ops.refs import cos_sim
    from dub.store import Workspace

    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--src", default="zh")
    a = ap.parse_args()
    cas = Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                         db_path=Path("/tmp/exp.sqlite")).cas  # fmt: skip
    rows = json.loads((a.job_dir / "units.json").read_text(encoding="utf-8"))
    refs = {
        r["speaker"]: r for r in json.loads((a.job_dir / "refs.json").read_text("utf-8"))["refs"]
    }
    lines = [r["placement"] for r in rows if r.get("placement")
             and r["placement"].get("variant") != "original"]  # fmt: skip

    def embed(key: str) -> list[float]:
        k16 = media.to_mono16k(AudioReq(audio_key=key), cas).audio_key
        return run_worker("tts", "ecapa_cluster", "audio.embed", {"audio_key": k16})["vectors"][0]

    ref_vec = {spk: embed(r["audio_key"]) for spk, r in refs.items()}

    def judge(take_key: str, text: str, spk: str) -> dict:
        tr = media.trim(TrimReq(audio_key=take_key), cas)
        k16 = media.to_mono16k(AudioReq(audio_key=tr.audio_key), cas).audio_key
        lid = run_worker(
            "asr", "whisper_large_v3", "asr.transcribe", {"audio_key": k16, "lang": "auto"}
        )
        heard = "".join(w["w"] for w in lid["words"])
        vi = run_worker(
            "asr", "phowhisper_large_ct2", "asr.backcheck", {"audio_key": k16, "lang": "vi"}
        )
        kept = plausible([Word.model_validate(w) for w in vi["words"]], tr.dur_ms, text,
                         "syllable", " ")  # fmt: skip
        return {
            "foreign": lid.get("lang") == a.src or bool(CJK.search(heard)),
            "cer": min(1.0, cer(text, " ".join(w.w for w in kept))),
            "sim": cos_sim(embed(tr.audio_key), ref_vec[spk]),
        }

    def synth(provider: str, env: str, p: dict, seed: int, params: dict | None = None) -> str:
        ref = refs[p["speaker"]]
        req = {"unit_id": f"accent-{p['unit_id']}", "text": p["text"], "lang": "vi",
               "speaker": p["speaker"], "seed": seed, "voice_key": ref["audio_key"],
               "voice_text": ref["text"]}  # fmt: skip
        if provider == "omnivoice":
            req["target_ms"] = p["clip_ms"] + 300
        return run_worker(env, provider, "tts.synthesize", req, params=params)["audio_key"]

    variants = {
        "omnivoice": lambda p: [synth("omnivoice", "tts", p, 0)],
        "omnivoice_g3": lambda p: [synth("omnivoice", "tts", p, 0, {"guidance_scale": 3.0})],
        "omnivoice_best3": lambda p: [synth("omnivoice", "tts", p, s) for s in (0, 1, 2)],
        "vieneu": lambda p: [synth("vieneu_v3_turbo", "tts2", p, 0)],
    }
    res: dict = {}
    for name, make in variants.items():
        scores = []
        for p in lines:
            takes = [judge(k, p["text"], p["speaker"]) for k in make(p)]
            best = min(takes, key=lambda t: (t["foreign"], t["cer"]))  # Vietnamese-sounding first
            scores.append(best)
        res[name] = {
            "lines": len(scores),
            "foreign_share": round(float(np.mean([s["foreign"] for s in scores])), 3),
            "cer_mean": round(float(np.mean([s["cer"] for s in scores])), 3),
            "cer_p90": round(float(np.percentile([s["cer"] for s in scores], 90)), 3),
            "sim_mean": round(float(np.mean([s["sim"] for s in scores])), 3),
        }
        print(name, json.dumps(res[name]), flush=True)
        save_json(data_dir() / "results" / "accent.json", res)


if __name__ == "__main__":
    main()
