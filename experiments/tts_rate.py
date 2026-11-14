"""Experiments E7 (natural speaking rate), E3 (safe squeeze range), E9 (16 vs 32 steps).

The timing ladder needs two numbers per target language:
- the speaking rate at which the TTS sounds natural (units/s: syllables, or characters for
  zh/ja/th), to predict how long each translation variant will be (E7);
- how far an exact-duration engine may be pushed from that rate before quality drops: the
  `safe_ratio` window (E3). Beyond it, time-stretching takes over, so we also compare
  OmniVoice at a forced length with rubberband stretching of the natural take.

OmniVoice clones a voice from a reference clip in ANOTHER language (as when dubbing), and
reads NTREX sentences at a grid of speaking rates. Quality signals:
- CER of a back-check ASR (Whisper large-v3; PhoWhisper for Vietnamese) - intelligibility;
- UTMOS (UTMOS22 strong, SpeechMOS) - naturalness. Trained on English: compare within a
  language only.
E9 repeats the natural rate with 16 diffusion steps instead of 32.

Steps (on the pod; the dubd workers must be running):
    envs/core/bin/python experiments/tts_rate.py gen --langs vi   # one process per language
    envs/core/bin/python experiments/tts_rate.py gen --langs en,es # group (they run in parallel)
    envs/tts/bin/python  experiments/tts_rate.py mos      # UTMOS scores
    envs/core/bin/python experiments/tts_rate.py report   # tables -> results/tts_rate.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (  # noqa: E402
    data_dir,
    documents,
    joiner_for,
    run_worker,
    save_json,
    worker_url,
)

REFS = {  # public sample prompts from the F5-TTS repository (MIT)
    "en_f": (
        "https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav",
        "Some call me nature, others call me mother nature.",
    ),
    "zh_m": (
        "https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/examples/basic/basic_ref_zh.wav",
        "对，这就是我，万人敬仰的太乙真人。",
    ),
}
PLAN = {  # target language -> reference voices (always from another language)
    "vi": ["en_f", "zh_m"],
    "en": ["zh_m"],
    "es": ["en_f"],
    "zh": ["en_f"],
    "ja": ["en_f"],
    "th": ["en_f"],
}
RATES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25]  # x the langpack's units_per_s
STRETCH = {("vi", "en_f")}  # (language, voice) pairs with a rubberband arm
N_SENT = 10
ASR_FOR = {"vi": "phowhisper_large_ct2"}


def _workspace():
    from dub.store import Workspace

    return Workspace.open(
        Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws", db_path=Path("/tmp/exp.sqlite")
    )


def _index_path(tag: str = "all") -> Path:
    return data_dir() / "results" / f"tts_index_{tag}.json"


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    for f in sorted((data_dir() / "results").glob("tts_index_*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    return rows


def sentences(lang: str) -> list[str]:
    """Mid-length sentences (about 2.5-6 s of speech)."""
    from dub.core.config import load_config
    from dub.ops.timing import count_units

    lp = load_config().langpack(lang)
    out = []
    for doc in documents(lang):
        for s in doc:
            sec = count_units(s, lp.speech_unit) / lp.units_per_s
            if 2.5 <= sec <= 6.0 and not any(c.isdigit() for c in s):
                out.append(s)
                break  # one per document: varied topics
        if len(out) == N_SENT:
            break
    return out


def gen(langs: list[str]) -> None:
    from dub.contracts.v1 import AudioReq, StretchReq, TrimReq
    from dub.core.config import load_config
    from dub.ops import media
    from dub.ops.align import cer
    from dub.ops.timing import count_units

    ws = _workspace()
    cas = ws.cas
    cfg = load_config()
    ref_keys = {}
    for name, (url, _text) in REFS.items():
        tmp = data_dir() / f"ref_{name}.wav"
        if not tmp.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "dub-experiments"})
            with urllib.request.urlopen(req, timeout=60) as r:
                tmp.write_bytes(r.read())
        # 16-bit mono 24 kHz, like the pipeline's references
        out = cas.tmp_path(".wav")
        media.run_ffmpeg(
            ["-i", str(tmp), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(out)]
        )
        ref_keys[name] = cas.put_file(out, move=True)

    rows = []

    def backcheck(lang: str, key: str, text: str) -> tuple[float, str]:
        a16 = media.to_mono16k(AudioReq(audio_key=key), cas)
        tl = run_worker(
            "asr",
            ASR_FOR.get(lang, "whisper_large_v3"),
            "asr.backcheck",
            {"audio_key": a16.audio_key, "lang": lang},
        )
        heard = (" " if joiner_for(lang) else "").join(w["w"] for w in tl["words"])
        return round(cer(text, heard), 4), heard

    def synth(base: dict, target_ms: int | None, steps: int, tag: str) -> dict:
        lang, voice = base["lang"], base["voice"]
        t0 = time.time()
        res = run_worker(
            "tts",
            "omnivoice",
            "tts.synthesize",
            {
                "unit_id": f"e3-{lang}-{voice}-{base['sent']}{tag}",
                "text": base["text"],
                "lang": lang,
                "speaker": voice,
                "voice_key": ref_keys[voice],
                "voice_text": REFS[voice][1],
                "target_ms": target_ms,
            },
            params={"num_step": steps},
        )
        sec = time.time() - t0
        trimmed = media.trim(TrimReq(audio_key=res["audio_key"]), cas)
        c, heard = backcheck(lang, trimmed.audio_key, base["text"])
        return {
            **base,
            "key": trimmed.audio_key,
            "clip_ms": res["dur_ms"],
            "voice_ms": trimmed.dur_ms,
            "lead_ms": trimmed.lead_ms,
            "tail_ms": trimmed.tail_ms,
            "cer": c,
            "heard": heard,
            "gen_s": round(sec, 3),
            "steps": steps,
        }

    tag = "-".join(langs)
    for lang in langs:
        voices = PLAN[lang]
        lp = cfg.langpack(lang)
        texts = sentences(lang)
        for voice in voices:
            for si, text in enumerate(texts):
                units = count_units(text, lp.speech_unit)
                base = {"lang": lang, "voice": voice, "sent": si, "text": text, "units": units}
                free = synth(base, None, 32, "-free")
                rows.append({**free, "arm": "free"})
                for r in RATES:
                    target = round(units / (r * lp.units_per_s) * 1000)
                    exact = synth(base, target, 32, f"-r{r}")
                    rows.append({**exact, "arm": "exact", "rate": r, "target_ms": target})
                    if (lang, voice) in STRETCH:
                        # rubberband the natural take to the same length
                        tempo = free["voice_ms"] / target
                        st = media.stretch(
                            StretchReq(
                                audio_key=free["key"],
                                rate=tempo,
                                engine="rubberband",
                                sample_rate=24000,
                            ),
                            cas,
                        )
                        c, heard = backcheck(lang, st.audio_key, text)
                        rows.append(
                            {
                                **base,
                                "arm": "stretch",
                                "rate": r,
                                "target_ms": target,
                                "key": st.audio_key,
                                "voice_ms": st.dur_ms,
                                "cer": c,
                                "heard": heard,
                                "tempo": round(tempo, 3),
                            }
                        )
                fast = synth(base, round(units / lp.units_per_s * 1000), 16, "-s16")
                rows.append({**fast, "arm": "steps16", "rate": 1.0})
            print(lang, voice, len(rows), flush=True)
            save_json(_index_path(tag), rows)
    save_json(_index_path(tag), rows)


def mos() -> None:
    import torch

    from dub.ops.audio_io import read_wav

    cas = _workspace().cas
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    model = model.cuda().eval()
    n = 0
    with torch.no_grad():
        for f in sorted((data_dir() / "results").glob("tts_index_*.json")):
            rows = json.loads(f.read_text(encoding="utf-8"))
            for r in rows:
                audio, sr = read_wav(cas.path(r["key"]))
                wave = torch.from_numpy(audio[:, 0]).unsqueeze(0).cuda()
                r["utmos"] = round(float(model(wave, sr).mean()), 3)
            save_json(f, rows)
            n += len(rows)
    print("scored", n)


def speed() -> None:
    """E9 timing, run alone on an idle GPU: 16 vs 32 steps on the same lines, interleaved.

    Uses the worker's own server-side time ("ms"), so HTTP and disk do not count."""
    from dub.core.config import load_config
    from dub.ops.timing import count_units
    from dub.worker.protocol import post_json

    cfg = load_config()
    out: dict = {}
    for lang in ("vi", "en", "zh"):
        lp = cfg.langpack(lang)
        times: dict[int, list[float]] = {16: [], 32: []}
        for si, text in enumerate(sentences(lang)):
            target = round(count_units(text, lp.speech_unit) / lp.units_per_s * 1000)
            for steps in (32, 16):
                body = {
                    "provider": "omnivoice",
                    "op": "tts.synthesize",
                    "params": {**cfg.providers["omnivoice"].params, "num_step": steps},
                    "request": {
                        "unit_id": f"e9-{lang}-{si}-{steps}",
                        "text": text,
                        "lang": lang,
                        "speaker": "S1",
                        "target_ms": target,
                        "seed": 7,
                    },
                }
                reply = post_json(f"{worker_url('tts')}/v1/run", body, timeout=600)
                times[steps].append(reply["ms"] / max(target, 1))  # ms of compute per ms of audio
        out[lang] = {str(k): {"rtf_p50": _med(v), "n": len(v)} for k, v in times.items()}
        print(lang, out[lang], flush=True)
    save_json(data_dir() / "results" / "tts_speed.json", out)


def _med(xs):
    return round(statistics.median(xs), 3) if xs else None


def _mean(xs):
    return round(statistics.fmean(xs), 3) if xs else None


def report() -> None:
    from dub.core.config import load_config

    cfg = load_config()
    rows = _load_rows()
    out: dict = {}
    for lang in PLAN:
        lp = cfg.langpack(lang)
        mine = [r for r in rows if r["lang"] == lang]
        free = [r for r in mine if r["arm"] == "free"]
        res: dict = {
            "prior_units_per_s": lp.units_per_s,
            "free": {
                "units_per_s": _med([r["units"] / (r["voice_ms"] / 1000) for r in free]),
                "cer": _mean([r["cer"] for r in free]),
                "utmos": _mean([r.get("utmos") for r in free if "utmos" in r]),
                "lead_ms": _med([r["lead_ms"] for r in free]),
                "tail_ms": _med([r["tail_ms"] for r in free]),
            },
        }
        for arm in ("exact", "stretch"):
            grid = {}
            for rate in RATES:
                sel = [r for r in mine if r["arm"] == arm and r["rate"] == rate]
                if sel:
                    grid[str(rate)] = {
                        "units_per_s": round(rate * lp.units_per_s, 2),
                        "cer": _mean([r["cer"] for r in sel]),
                        "cer_bad": sum(1 for r in sel if r["cer"] > 0.15),
                        "utmos": _mean([r["utmos"] for r in sel if "utmos" in r]),
                        "n": len(sel),
                    }
            if grid:
                res[arm] = grid
        s16 = [r for r in mine if r["arm"] == "steps16"]
        s32 = [r for r in mine if r["arm"] == "exact" and r["rate"] == 1.0]
        res["steps"] = {
            "16": {
                "cer": _mean([r["cer"] for r in s16]),
                "utmos": _mean([r.get("utmos") for r in s16 if "utmos" in r]),
                "gen_s": _med([r["gen_s"] for r in s16]),
            },
            "32": {
                "cer": _mean([r["cer"] for r in s32]),
                "utmos": _mean([r.get("utmos") for r in s32 if "utmos" in r]),
                "gen_s": _med([r["gen_s"] for r in s32]),
            },
        }
        out[lang] = res
        print(lang, json.dumps(res, ensure_ascii=False))
    save_json(data_dir() / "results" / "tts_rate.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "mos", "report", "speed"])
    ap.add_argument("--langs", default=",".join(PLAN))
    a = ap.parse_args()
    if a.step == "gen":
        gen(a.langs.split(","))
    else:
        {"mos": mos, "report": report, "speed": speed}[a.step]()


if __name__ == "__main__":
    main()
