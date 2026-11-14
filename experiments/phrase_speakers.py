"""Who says each line? Diarization scored phrase by phrase against hand labels (M8).

On a zh drama the dub gave one voice to a whole hiring interview (0-15 s: "How many months?"
"Three." "Boy or girl?" "Boy." ...): ECAPA windows of 1.5 s straddle turns that short. Here
every burned-in subtitle cue is a phrase whose speaker was labelled by hand from the script
(experiments/golden/*.speakers.json; '?' cues are left out of the scores). Compared:
- `diar_T`: the `ecapa_cluster` diarizer at threshold T (speaker of the turn under the cue);
- `phrase_T`: the cues themselves embedded (ECAPA, on the separated vocals) and clustered
  (average linkage, cosine distance T); cues under `--min-ms` take the nearest cluster;
- `reassign_T`: the diarizer's speakers at T, then every cue moved to the nearest centroid;
- `job`: the speaker of the dubbed unit under each cue in a finished job (`--units`).
Scores, on the labelled cues:
- B-cubed precision / recall / F1 (no mapping of found to true speakers needed): precision
  falls when one voice carries several people, recall when one person gets several voices;
- turn changes: of consecutive labelled cues (gap < 2 s) where the speaker changes, how many
  change voice (recall), and of voice changes, how many are real (precision).
Also printed: how well ECAPA tells each pair of people apart (mean cosine within vs between).

Pod (workers running):
    envs/core/bin/python experiments/phrase_speakers.py SRC_KEY LABELS.json [--units units.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, run_worker, save_json  # noqa: E402

CHANGE_GAP_MS = 2000


def _ws():
    from dub.store import Workspace

    return Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                          db_path=Path("/tmp/exp.sqlite"))  # fmt: skip


def bcubed(truth: list[str], found: list[str]) -> tuple[float, float, float]:
    n = len(truth)
    p = r = 0.0
    for i in range(n):
        same_found = [j for j in range(n) if found[j] == found[i]]
        same_true = [j for j in range(n) if truth[j] == truth[i]]
        p += sum(truth[j] == truth[i] for j in same_found) / len(same_found)
        r += sum(found[j] == found[i] for j in same_true) / len(same_true)
    p, r = p / n, r / n
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def changes(cues: list[dict], found: list[str]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for a, b in itertools.pairwise(range(len(cues))):
        ca, cb = cues[a], cues[b]
        if "?" in (ca["who"], cb["who"]) or cb["start_ms"] - ca["end_ms"] > CHANGE_GAP_MS:
            continue
        true_change = ca["who"] != cb["who"]
        found_change = found[a] != found[b]
        tp += true_change and found_change
        fp += found_change and not true_change
        fn += true_change and not found_change
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def score(cues: list[dict], found: list[str]) -> dict:
    keep = [i for i, c in enumerate(cues) if c["who"] != "?"]
    bp, br, bf = bcubed([cues[i]["who"] for i in keep], [found[i] for i in keep])
    cp, cr, cf = changes(cues, found)
    return {"voices": len(set(found)), "b3_p": round(bp, 3), "b3_r": round(br, 3),
            "b3_f": round(bf, 3), "chg_p": round(cp, 3), "chg_r": round(cr, 3),
            "chg_f": round(cf, 3)}  # fmt: skip


def under(turns: list[dict], c: dict) -> str:
    best, ov = "-", 0
    for t in turns:
        o = min(t["end_ms"], c["end_ms"]) - max(t["start_ms"], c["start_ms"])
        if o > ov:
            best, ov = t["speaker"], o
    return best


def average_linkage(vecs: np.ndarray, t: float) -> np.ndarray:
    """Agglomerative clustering, average linkage on cosine distance, cut at distance t
    (numpy only: the core env has no scipy)."""
    n = len(vecs)
    groups = [[i] for i in range(n)]
    d = 1.0 - vecs @ vecs.T
    np.fill_diagonal(d, np.inf)
    while len(groups) > 1:
        k = int(np.argmin(d))
        a, b = divmod(k, len(groups))
        if d[a, b] > t:
            break
        a, b = min(a, b), max(a, b)
        na, nb = len(groups[a]), len(groups[b])
        merged = (d[a] * na + d[b] * nb) / (na + nb)
        d[a, :], d[:, a] = merged, merged
        d[a, a] = np.inf
        d = np.delete(np.delete(d, b, axis=0), b, axis=1)
        groups[a] += groups.pop(b)
    labels = np.zeros(n, dtype=int)
    for g, members in enumerate(groups):
        labels[members] = g + 1
    return labels


def cluster(vecs: np.ndarray, long_idx: list[int], t: float) -> list[int]:
    labels = np.zeros(len(vecs), dtype=int)
    lab = average_linkage(vecs[long_idx], t)
    cents = {k: vecs[long_idx][lab == k].mean(axis=0) for k in np.unique(lab)}
    keys = sorted(cents)
    cmat = np.stack([cents[k] / np.linalg.norm(cents[k]) for k in keys])
    for i in range(len(vecs)):
        labels[i] = keys[int(np.argmax(cmat @ vecs[i]))]
    for j, i in enumerate(long_idx):
        labels[i] = lab[j]
    return labels.tolist()


def main() -> None:
    from dub.contracts.v1 import AudioReq, DecodeReq
    from dub.ops import media

    ap = argparse.ArgumentParser()
    ap.add_argument("src_key")
    ap.add_argument("labels", type=Path)
    ap.add_argument("--units", type=Path)
    ap.add_argument("--min-ms", type=int, default=600)
    a = ap.parse_args()
    cues = json.loads(a.labels.read_text(encoding="utf-8"))["cues"]
    cas = _ws().cas
    info = media.decode(DecodeReq(src_key=a.src_key), cas)
    st = run_worker("sep", "mel_roformer", "audio.separate", {"audio_key": info.a48k_key})
    v16 = media.to_mono16k(AudioReq(audio_key=st["vocals_key"]), cas).audio_key
    res: dict = {}

    if a.units:
        rows = json.loads(a.units.read_text(encoding="utf-8"))
        turns = [{"speaker": r["unit"]["speaker"], "start_ms": r["unit"]["start_ms"],
                  "end_ms": r["unit"]["end_ms"]} for r in rows]  # fmt: skip
        res["job"] = score(cues, [under(turns, c) for c in cues])

    diar_turns = {}
    for t in (0.85, 0.8, 0.75, 0.7):
        d = run_worker("tts", "ecapa_cluster", "audio.diarize", {"audio_key": v16},
                       params={"threshold": t})  # fmt: skip
        diar_turns[t] = d["turns"]
        res[f"diar_{t}"] = score(cues, [under(d["turns"], c) for c in cues])
        print(f"diar_{t}", json.dumps(res[f"diar_{t}"]), flush=True)

    spans = [(c["start_ms"], c["end_ms"]) for c in cues]
    vecs = np.asarray(run_worker("tts", "ecapa_cluster", "audio.embed",
                                 {"audio_key": v16, "spans": spans})["vectors"])  # fmt: skip
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    long_idx = [i for i, c in enumerate(cues) if c["end_ms"] - c["start_ms"] >= a.min_ms]
    for t in (0.5, 0.6, 0.7, 0.8):
        res[f"phrase_{t}"] = score(cues, [str(x) for x in cluster(vecs, long_idx, t)])

    for t in (0.85, 0.75):
        base = [under(diar_turns[t], c) for c in cues]
        names = sorted(set(base) - {"-"})
        cents = []
        for n in names:
            idx = [i for i in long_idx if base[i] == n] or [
                i for i in range(len(cues)) if base[i] == n
            ]
            c = vecs[idx].mean(axis=0)
            cents.append(c / np.linalg.norm(c))
        cm = np.stack(cents)
        res[f"reassign_{t}"] = score(cues, [names[int(np.argmax(cm @ v))] for v in vecs])

    for k, v in res.items():
        print(f"{k:14} {json.dumps(v)}")

    # how far apart are the people, for ECAPA (long cues only)
    who = sorted({cues[i]["who"] for i in long_idx} - {"?"})
    sims = vecs @ vecs.T
    sep = {}
    for x, y in itertools.combinations_with_replacement(who, 2):
        ix = [i for i in long_idx if cues[i]["who"] == x]
        iy = [i for i in long_idx if cues[i]["who"] == y]
        vals = [sims[i, j] for i in ix for j in iy if i != j]
        if len(ix) >= 2 and len(iy) >= 2 and vals:
            sep[f"{x}-{y}"] = round(float(np.mean(vals)), 3)
    print("mean cosine (same person on the diagonal):", json.dumps(sep))
    res["cosine"] = sep
    save_json(data_dir() / "results" / "phrase_speakers.json", res)


if __name__ == "__main__":
    main()
