"""Experiment E8: who speaks when. ecapa_cluster (with a threshold sweep, and with the true
number of speakers given) against the single-speaker stand-in.

Material:
- drama4: the 4-person family scene of drama.json (synthetic voices, quick turn-taking);
- interview2: the 2-voice SAPI interview clip with known line onsets (music bed);
- ami4: AMI meeting ES2004a, first 10 minutes, real 4-person meeting (CC BY 4.0), with the
  reference RTTM of pyannote's AMI setup.

Metrics, on 10 ms frames, after the best one-to-one mapping of found speakers to true ones:
- confusion: share of frames where both truth and system hear speech but disagree on who;
- missed / false alarm: speech the system did not attribute / attributed where there is none;
- speakers found vs true.
Unit-level speaker swaps are measured on full pipeline runs (drama and interview).

On the pod:  envs/core/bin/python experiments/e8_diarization.py --interview-key KEY
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, run_worker, save_json  # noqa: E402

AMI_WAV = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav"
AMI_RTTM = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/ES2004a.rttm"
AMI_SECONDS = 600
THRESHOLDS = [0.7, 0.75, 0.8, 0.85, 0.9]
MIN_SHARES = [0.02, 0.05]
FRAME = 10


def _get(url: str, path: Path) -> Path:
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "dub-experiments"})
        with urllib.request.urlopen(req, timeout=600) as r:
            path.write_bytes(r.read())
    return path


def _frames(turns: list[tuple[str, int, int]], n: int) -> list[set[str]]:
    out: list[set[str]] = [set() for _ in range(n)]
    for spk, s, e in turns:
        for f in range(max(0, s // FRAME), min(n, e // FRAME)):
            out[f].add(spk)
    return out


def score(truth: list[tuple[str, int, int]], hyp: list[tuple[str, int, int]], end_ms: int) -> dict:
    n = end_ms // FRAME
    tf, hf = _frames(truth, n), _frames(hyp, n)
    tspk = sorted({t[0] for t in truth})
    hspk = sorted({h[0] for h in hyp})
    # overlap matrix, then the best mapping (brute force: <= 6 speakers)
    ov = {(h, t): 0 for h in hspk for t in tspk}
    for a, b in zip(hf, tf, strict=True):
        for h in a:
            for t in b:
                ov[(h, t)] += 1
    best, best_map = -1, {}
    k = min(len(hspk), len(tspk))
    for hs in itertools.permutations(hspk, k):
        for ts in itertools.combinations(tspk, k):
            for perm in itertools.permutations(ts):
                v = sum(ov[(h, t)] for h, t in zip(hs, perm, strict=True))
                if v > best:
                    best, best_map = v, dict(zip(hs, perm, strict=True))
        if len(hspk) > 6:
            break
    both = conf = miss = fa = 0
    for a, b in zip(hf, tf, strict=True):
        if b and a:
            both += 1
            mapped = {best_map.get(h) for h in a}
            if not (mapped & b):
                conf += 1
        elif b:
            miss += 1
        elif a:
            fa += 1
    speech = sum(1 for b in tf if b)
    return {
        "speakers_true": len(tspk),
        "speakers_found": len(hspk),
        "confusion": round(conf / max(both, 1), 4),
        "missed": round(miss / max(speech, 1), 4),
        "false_alarm": round(fa / max(speech, 1), 4),
    }


def _vocals16(cas, key: str) -> tuple[str, int]:
    from dub.contracts.v1 import AudioReq, DecodeReq
    from dub.ops import media

    info = media.decode(DecodeReq(src_key=key), cas)
    stems = run_worker("sep", "mel_roformer", "audio.separate", {"audio_key": info.a48k_key})
    v16 = media.to_mono16k(AudioReq(audio_key=stems["vocals_key"]), cas)
    return v16.audio_key, info.duration_ms


def main() -> None:
    from dub.ops.media import run_ffmpeg
    from dub.store import Workspace

    ap = argparse.ArgumentParser()
    ap.add_argument("--interview-key", required=True, help="CAS key of the SAPI interview audio")
    a = ap.parse_args()
    ws = Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                        db_path=Path("/tmp/exp.sqlite"))  # fmt: skip
    cas = ws.cas
    sets = {}
    drama = json.loads((data_dir() / "results" / "drama.json").read_text(encoding="utf-8"))
    sets["drama4"] = (drama["audio_key"], [(x["speaker"], x["onset_ms"], x["offset_ms"])
                                           for x in drama["lines"]])  # fmt: skip
    truth_iv = json.loads((Path(__file__).parent / "interview_truth.json").read_text("utf-8"))
    sets["interview2"] = (a.interview_key, [(x["speaker"], x["onset_ms"], x["offset_ms"])
                                            for x in truth_iv])  # fmt: skip
    wav = _get(AMI_WAV, data_dir() / "ES2004a.Mix-Headset.wav")
    rttm = _get(AMI_RTTM, data_dir() / "ES2004a.rttm").read_text(encoding="utf-8").splitlines()
    cut = cas.tmp_path(".wav")
    run_ffmpeg(["-i", str(wav), "-t", str(AMI_SECONDS), "-ac", "2", "-ar", "48000", str(cut)])
    ami_truth = []
    for ln in rttm:
        f = ln.split()
        s, d = float(f[3]), float(f[4])
        if s < AMI_SECONDS:
            ami_truth.append((f[7], round(s * 1000), round(min(s + d, AMI_SECONDS) * 1000)))
    sets["ami4"] = (cas.put_file(cut, move=True), ami_truth)

    results = {}
    for name, (key, truth) in sets.items():
        v16, dur = _vocals16(cas, key)
        n_true = len({t[0] for t in truth})
        row = {"single_speaker": score(truth, [("S1", 0, dur)], dur)}
        for th in THRESHOLDS:
            for ms in MIN_SHARES:
                d = run_worker("tts", "ecapa_cluster", "audio.diarize", {"audio_key": v16},
                               params={"threshold": th, "min_share": ms})  # fmt: skip
                hyp = [(t["speaker"], t["start_ms"], t["end_ms"]) for t in d["turns"]]
                row[f"ecapa@{th}/{ms}"] = score(truth, hyp, dur)
        d = run_worker("tts", "ecapa_cluster", "audio.diarize",
                       {"audio_key": v16, "speakers_hint": n_true})  # fmt: skip
        row["ecapa@k=true"] = score(truth, [(t["speaker"], t["start_ms"], t["end_ms"])
                                            for t in d["turns"]], dur)  # fmt: skip
        results[name] = row
        for k, v in row.items():
            print(f"{name:<11} {k:<15} {v}", flush=True)
    save_json(data_dir() / "results" / "e8.json", results)


if __name__ == "__main__":
    main()
