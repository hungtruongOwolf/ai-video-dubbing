"""Why does OmniVoice leave silence inside the duration we ask for, and how do we stop it?

Found on the 60-minute speech (draft, en -> vi): after trimming, the voice is only 90% of the
requested length (p50; 76% for lines under 2 s), with ~340 ms of silence at the start and
~100 ms at the end. The speech is squeezed into the rest: 4.8 syllables/s (p50, p90 6.1)
where 4.2 was asked. Suspect: the edges of the reference clip (the model continues the
prompt, silence included).

Per reference voice (the five F5-TTS recordings, plus any keys passed with --ref KEY:TEXT),
twelve Vietnamese sentences, three ways:
- as is: the reference clip unchanged;
- tight: the reference trimmed to its voice (20 ms margin);
- tight + pad: tight reference, and the requested length grows by the silence measured on
  the "tight" takes of that voice (a per-voice correction).
Reported: lead and tail silence of the takes, voice length / requested length, speaking rate.

Pod (workers running):  envs/core/bin/python experiments/tts_silence.py [--ref KEY:TEXT ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, documents, run_worker, save_json  # noqa: E402

N_SENT = 12


def _ws():
    from dub.store import Workspace

    return Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                          db_path=Path("/tmp/exp.sqlite"))  # fmt: skip


def _sentences() -> list[str]:
    from dub.ops.timing import count_units

    out = []
    for doc in documents("vi"):
        for s in doc:
            n = count_units(s, "syllable")
            if 8 <= n <= 30 and not any(c.isdigit() for c in s):
                out.append(s)
                break
        if len(out) == N_SENT:
            break
    return out


def _edges_ms(key: str, cas) -> tuple[int, int, int]:
    """(lead silence, voice length, tail silence) of a clip, as our trim sees it."""
    from dub.contracts.v1 import TrimReq
    from dub.ops import media

    t = media.trim(TrimReq(audio_key=key), cas)
    return t.lead_ms, t.dur_ms, t.tail_ms


def _tight(key: str, cas) -> str:
    from dub.contracts.v1 import TrimReq
    from dub.ops import media

    return media.trim(TrimReq(audio_key=key, pad_ms=20), cas).audio_key


def main() -> None:
    from dub.contracts.v1 import AudioReq
    from dub.core.config import load_config
    from dub.ops import media
    from dub.ops.timing import count_units

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="append", default=[], help="CAS_KEY:transcript")
    a = ap.parse_args()
    cas = _ws().cas
    cfg = load_config()
    rate = cfg.langpack("vi").units_per_s
    voices: dict[str, tuple[str, str]] = {}
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
        voices[f.stem] = (key, text)
    for item in a.ref:
        key, _, text = item.partition(":")
        voices[f"ref_{key[:8]}"] = (key, text)

    sents = _sentences()
    res: dict = {}
    for vname, (key, text) in voices.items():
        ref_lead, _ref_voice, ref_tail = _edges_ms(key, cas)
        tight = _tight(key, cas)
        row: dict = {"ref_lead_ms": ref_lead, "ref_tail_ms": ref_tail}
        pad = 0
        for mode in ("as_is", "tight", "tight_pad"):
            ref = key if mode == "as_is" else tight
            leads, tails, ratios, rates = [], [], [], []
            for si, s in enumerate(sents):
                n = count_units(s, "syllable")
                target = round(n / rate * 1000) + (pad if mode == "tight_pad" else 0)
                out = run_worker("tts", "omnivoice", "tts.synthesize", {
                    "unit_id": f"sil-{vname}-{si}", "text": s, "lang": "vi", "speaker": vname,
                    "voice_key": ref, "voice_text": text, "target_ms": target,
                }, params={"num_step": 16})  # fmt: skip
                lead, voice, tail = _edges_ms(out["audio_key"], cas)
                leads.append(lead)
                tails.append(tail)
                ratios.append(voice / (target - (pad if mode == "tight_pad" else 0)))
                rates.append(n / max(voice / 1000, 0.1))
            row[mode] = {
                "lead_ms_p50": int(np.median(leads)), "tail_ms_p50": int(np.median(tails)),
                "voice_over_planned_p50": round(float(np.median(ratios)), 3),
                "voice_over_planned_p10": round(float(np.percentile(ratios, 10)), 3),
                "syl_per_s_p50": round(float(np.median(rates)), 2),
            }  # fmt: skip
            if mode == "tight":
                pad = int(np.median(leads) + np.median(tails))
                row["pad_ms"] = pad
        res[vname] = row
        print(vname, json.dumps(row), flush=True)
        save_json(data_dir() / "results" / "tts_silence.json", res)


if __name__ == "__main__":
    main()
