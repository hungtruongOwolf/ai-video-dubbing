"""Experiment E1 (+E2): which TTS engine for Vietnamese, cloning voices of other languages?

OmniVoice (incumbent; exact length at the vi natural rate) against VieNeu-TTS v3 Turbo
(free-running), on the same lines and voices:
- 5 reference voices, none Vietnamese (the drama's four F5-TTS voices and F5-TTS's English
  voice): the dubbing situation, where the reference is the original speaker (E2: accent
  and timbre carry-over across languages);
- 12 mid-length Vietnamese NTREX sentences.
Per take: back-check CER (PhoWhisper), speaker similarity to the reference clip (ECAPA
cosine), UTMOS (naturalness, English-trained: compare engines, not absolute values),
real-time factor on the worker, and VieNeu's own speaking rate.

Pod:  envs/core/bin/python experiments/e1_tts.py gen
      envs/tts/bin/python  experiments/e1_tts.py mos
      envs/core/bin/python experiments/e1_tts.py report
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, documents, run_worker, save_json, worker_url  # noqa: E402

F5 = "https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/examples"
VOICES = {
    "country_m": f"{F5}/multi/country.flac",
    "main_f": f"{F5}/multi/main.flac",
    "town_m": f"{F5}/multi/town.flac",
    "zh_voice": f"{F5}/basic/basic_ref_zh.wav",
    "en_f": f"{F5}/basic/basic_ref_en.wav",
}
ENGINES = {"omnivoice": "tts", "vieneu_v3_turbo": "tts2"}
N_SENT = 12


def _ws():
    from dub.store import Workspace

    return Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                          db_path=Path("/tmp/exp.sqlite"))  # fmt: skip


def _index() -> Path:
    return data_dir() / "results" / "e1_index.json"


def _sentences() -> list[str]:
    from dub.ops.timing import count_units

    out = []
    for doc in documents("vi"):
        for s in doc:
            n = count_units(s, "syllable")
            if 12 <= n <= 28 and not any(c.isdigit() for c in s):
                out.append(s)
                break
        if len(out) == N_SENT:
            break
    return out


def gen() -> None:
    from dub.contracts.v1 import AudioReq, TrimReq
    from dub.core.config import load_config
    from dub.ops import media
    from dub.ops.align import cer
    from dub.ops.timing import count_units
    from dub.worker.protocol import post_json

    cas = _ws().cas
    cfg = load_config()
    refs = {}
    for name, url in VOICES.items():
        raw = data_dir() / f"e1_{name}{Path(url).suffix}"
        if not raw.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "dub-experiments"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw.write_bytes(r.read())
        out = cas.tmp_path(".wav")
        media.run_ffmpeg(["-i", str(raw), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
                          str(out)])  # fmt: skip
        key = cas.put_file(out, move=True)
        k16 = media.to_mono16k(AudioReq(audio_key=key), cas).audio_key
        text = " ".join(w["w"] for w in run_worker(
            "asr", "whisper_large_v3", "asr.transcribe", {"audio_key": k16, "lang": "auto"}
        )["words"])  # fmt: skip
        vec = run_worker("tts", "ecapa_cluster", "audio.embed", {"audio_key": k16})["vectors"][0]
        refs[name] = {"key": key, "text": text, "vec": vec}
        print("voice", name, text[:50], flush=True)

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    rows = []
    for si, text in enumerate(_sentences()):
        units = count_units(text, "syllable")
        for vname, ref in refs.items():
            for engine, env in ENGINES.items():
                m = cfg.providers[engine]
                req = {
                    "unit_id": f"e1-{si}-{vname}",
                    "text": text,
                    "lang": "vi",
                    "speaker": vname,
                    "voice_key": ref["key"],
                    "voice_text": ref["text"],
                }
                if m.caps.duration_control == "exact":
                    rate = cfg.langpack("vi").units_per_s
                    req["target_ms"] = round(units / rate * 1000)
                reply = post_json(f"{worker_url(env)}/v1/run", {
                    "provider": engine, "op": "tts.synthesize", "params": m.params, "request": req,
                }, timeout=900)  # fmt: skip
                res = reply["result"]
                trimmed = media.trim(TrimReq(audio_key=res["audio_key"]), cas)
                k16 = media.to_mono16k(AudioReq(audio_key=trimmed.audio_key), cas).audio_key
                tl = run_worker("asr", "phowhisper_large_ct2", "asr.backcheck",
                                {"audio_key": k16, "lang": "vi"})  # fmt: skip
                heard = " ".join(w["w"] for w in tl["words"])
                vec = run_worker("tts", "ecapa_cluster", "audio.embed",
                                 {"audio_key": k16})["vectors"][0]  # fmt: skip
                rows.append({
                    "sent": si, "voice": vname, "engine": engine, "text": text, "heard": heard,
                    "key": trimmed.audio_key, "voice_ms": trimmed.dur_ms, "units": units,
                    "cer": round(cer(text, heard), 4), "sim": round(cos(vec, ref["vec"]), 4),
                    "rtf": round(reply["ms"] / max(res["dur_ms"], 1), 4),
                })  # fmt: skip
        print("sentence", si, len(rows), flush=True)
        save_json(_index(), rows)


def mos() -> None:
    import torch

    from dub.ops.audio_io import read_wav

    cas = _ws().cas
    rows = json.loads(_index().read_text(encoding="utf-8"))
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    model = model.cuda().eval()
    with torch.no_grad():
        for r in rows:
            audio, sr = read_wav(cas.path(r["key"]))
            r["utmos"] = round(
                float(model(torch.from_numpy(audio[:, 0]).unsqueeze(0).cuda(), sr)), 3
            )
    save_json(_index(), rows)
    print("scored", len(rows))


def report() -> None:
    rows = json.loads(_index().read_text(encoding="utf-8"))
    out = {}
    for engine in ENGINES:
        sel = [r for r in rows if r["engine"] == engine]
        if not sel:
            continue
        out[engine] = {
            "takes": len(sel),
            "cer_mean": round(statistics.fmean(r["cer"] for r in sel), 4),
            "cer_over_15pct": sum(1 for r in sel if r["cer"] > 0.15),
            "sim_mean": round(statistics.fmean(r["sim"] for r in sel), 4),
            "utmos_mean": round(statistics.fmean(r["utmos"] for r in sel if "utmos" in r), 3)
            if any("utmos" in r for r in sel)
            else None,
            "rtf_p50": round(statistics.median(r["rtf"] for r in sel), 4),
            "syllables_per_s": round(
                statistics.median(r["units"] / (r["voice_ms"] / 1000) for r in sel), 2
            ),  # fmt: skip
            "by_voice": {
                v: {
                    "cer": round(statistics.fmean(r["cer"] for r in sel if r["voice"] == v), 4),
                    "sim": round(statistics.fmean(r["sim"] for r in sel if r["voice"] == v), 4),
                }
                for v in VOICES
            },
        }
    save_json(data_dir() / "results" / "e1.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "mos", "report"])
    a = ap.parse_args()
    {"gen": gen, "mos": mos, "report": report}[a.step]()


if __name__ == "__main__":
    main()
