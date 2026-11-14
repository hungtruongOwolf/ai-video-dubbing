"""Separation quality and speed per setting (Mel-Band RoFormer through audio-separator).

Questions (raised by the code review of 2026-11-14):
- the mix was loaded at 48 kHz, but the model was trained at 44.1 kHz: does loading at
  44.1 kHz separate better?
- the separator ran in fp32: does autocast or native fp16 keep the quality and save time?

Quality: synthetic mixes with known stems. Clean speech (the five F5-TTS reference
recordings, joined, 60 s) over three public-domain pieces from Wikimedia Commons (piano,
orchestra, electronic), each at two levels (music 6 dB and 12 dB below the speech). SI-SDR of
the separated vocals against the clean speech, and of the instrumental against the music.
Speed: one separation of the 60-minute test speech per setting.

Pod (workers running):  envs/core/bin/python experiments/sep_quality.py run --long-key KEY
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, run_worker, save_json  # noqa: E402

MUSIC = {
    "piano": "Schubert - Piano Sonata No. 13 in A major, D664 - II. Andante (Paul Pitman).flac",
    "orchestra": "Anthem of Kamchatka Krai (orchestral instrumental version).ogg",
    "electronic": "Kevin MacLeod - Long Trail.ogg",
}
SETTINGS = {
    "48k_fp32": {},
    "44k_fp32": {"sample_rate": 44100},
    "44k_autocast": {"sample_rate": 44100, "precision": "autocast"},
    "44k_fp16": {"sample_rate": 44100, "precision": "fp16"},
}
SR = 48000
SECONDS = 60


def _commons_url(title: str) -> str:
    import urllib.parse

    params = {"action": "query", "titles": f"File:{title}", "prop": "imageinfo",
              "iiprop": "url", "format": "json"}  # fmt: skip
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{q}",
                                 headers={"User-Agent": "dub-experiments/0.1"})  # fmt: skip
    page = next(iter(json.load(urllib.request.urlopen(req, timeout=60))["query"]["pages"].values()))
    return page["imageinfo"][0]["url"]


def _load(path: Path, cas) -> np.ndarray:
    from dub.ops.audio_io import read_wav
    from dub.ops.media import run_ffmpeg

    tmp = cas.tmp_path(".wav")
    run_ffmpeg(["-i", str(path), "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(tmp)])
    audio, _ = read_wav(tmp)
    tmp.unlink()
    return audio[:, 0].astype(np.float64)


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    target = (est @ ref) / (ref @ ref + 1e-12) * ref
    noise = est - target
    return round(float(10 * np.log10((target @ target) / (noise @ noise + 1e-12))), 2)


def make(cas) -> list[dict]:
    """The mixes (and their clean stems) as CAS files."""
    from dub.ops.audio_io import write_wav

    root = data_dir() / "sepq"
    root.mkdir(parents=True, exist_ok=True)
    speech_parts = []
    for f in sorted(data_dir().glob("e1_*.*")):
        if f.suffix in (".wav", ".flac"):
            speech_parts += [_load(f, cas), np.zeros(SR * 3 // 10)]
    speech = np.concatenate(speech_parts)
    speech = np.tile(speech, SECONDS * SR // len(speech) + 1)[: SECONDS * SR]
    speech_rms = np.sqrt(np.mean(speech[np.abs(speech) > 1e-3] ** 2))
    out = []
    for name, src in MUSIC.items():
        raw = root / f"{name}{Path(src).suffix}"
        if not raw.exists():
            url = _commons_url(src)
            req = urllib.request.Request(url, headers={"User-Agent": "dub-experiments/0.1"})
            raw.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        music = _load(raw, cas)[SR * 20 : SR * (20 + SECONDS)]
        music_rms = np.sqrt(np.mean(music**2))
        for db in (6, 12):
            m = music * (speech_rms / music_rms) * 10 ** (-db / 20)
            mix = speech + m
            peak = max(1.0, float(np.abs(mix).max()) / 0.95)
            keys = {}
            for kind, sig in (("mix", mix / peak), ("speech", speech / peak), ("bed", m / peak)):
                tmp = cas.tmp_path(".wav")
                write_wav(tmp, np.stack([sig, sig], axis=1).astype(np.float32), SR)
                keys[kind] = cas.put_file(tmp, move=True)
            out.append({"music": name, "db_below": db, **keys})
    return out


def run(long_key: str | None) -> None:
    import os

    from dub.ops.audio_io import read_wav
    from dub.store import Workspace

    ws = Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                        db_path=Path("/tmp/exp.sqlite"))  # fmt: skip
    cas = ws.cas
    mixes = make(cas)
    out = data_dir() / "results" / "sep_quality.json"
    res: dict = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for label, params in SETTINGS.items():
        if label in res:
            continue  # measured by an earlier (interrupted) run
        rows = []
        for m in mixes:
            st = run_worker("sep", "mel_roformer", "audio.separate", {"audio_key": m["mix"]},
                            params=params)  # fmt: skip
            voc = read_wav(cas.path(st["vocals_key"]))[0].mean(axis=1)
            ins = read_wav(cas.path(st["instrumental_key"]))[0].mean(axis=1)
            speech = read_wav(cas.path(m["speech"]))[0][:, 0]
            music = read_wav(cas.path(m["bed"]))[0][:, 0]
            for k in (st["vocals_key"], st["instrumental_key"]):
                cas.path(k).unlink(missing_ok=True)  # experiment output: keep the quota free
            rows.append({"music": m["music"], "db_below": m["db_below"],
                         "vocals_si_sdr": si_sdr(voc, speech),
                         "bed_si_sdr": si_sdr(ins, music)})  # fmt: skip
        res[label] = {
            "vocals_si_sdr_mean": round(float(np.mean([r["vocals_si_sdr"] for r in rows])), 2),
            "bed_si_sdr_mean": round(float(np.mean([r["bed_si_sdr"] for r in rows])), 2),
            "rows": rows,
        }
        if long_key:
            t = time.time()
            st = run_worker("sep", "mel_roformer", "audio.separate", {"audio_key": long_key},
                            params=params)  # fmt: skip
            res[label]["seconds_60min"] = round(time.time() - t, 1)
            for k in (st["vocals_key"], st["instrumental_key"]):
                cas.path(k).unlink(missing_ok=True)  # 1.4 GB per hour of audio
        print(label, {k: v for k, v in res[label].items() if k != "rows"}, flush=True)
        save_json(out, res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run"])
    ap.add_argument("--long-key", default=None, help="48 kHz audio key of a long video (speed)")
    a = ap.parse_args()
    run(a.long_key)


if __name__ == "__main__":
    main()
