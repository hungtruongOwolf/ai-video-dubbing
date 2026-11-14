"""Speaker diarization helpers (pure numpy): windows over speech, labels -> speaker turns.

The model side (speaker embeddings, clustering) lives in the provider; these steps are
shared and testable without a GPU:

1. `windows`: 1.5 s windows every 0.75 s over the VAD speech regions (short regions give one
   window of their own length).
2. the provider embeds each window and clusters the embeddings;
3. `absorb_small`: a cluster holding very little speech (a laugh, a cough, a noise) is folded
   into the nearest real speaker;
4. `turns_from_windows`: every 10 ms frame of speech takes the label of the window centred
   nearest to it; runs of one label become turns, and a turn shorter than `min_turn_ms`
   between two turns of the same speaker is absorbed by them.

Speakers are named S1, S2... in order of first appearance.
"""

from __future__ import annotations

import numpy as np

from dub.contracts.v1 import Turn

FRAME_MS = 10


def windows(
    regions: list[tuple[int, int]], win_ms: int = 1500, hop_ms: int = 750, min_ms: int = 400
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in regions:
        if e - s < min_ms:
            continue
        if e - s <= win_ms:
            out.append((s, e))
            continue
        t = s
        while t + win_ms < e:
            out.append((t, t + win_ms))
            t += hop_ms
        out.append((max(s, e - win_ms), e))
    return out


def centroids(embs: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for lab in np.unique(labels):
        c = embs[labels == lab].mean(axis=0)
        out[int(lab)] = c / (np.linalg.norm(c) + 1e-9)
    return out


def absorb_small(
    embs: np.ndarray,
    labels: np.ndarray,
    wins: list[tuple[int, int]],
    min_share: float = 0.02,
    min_windows: int = 3,
) -> np.ndarray:
    """Fold clusters with too little speech into the closest remaining cluster."""
    labels = labels.copy()
    total = sum(e - s for s, e in wins) or 1
    while True:
        labs = np.unique(labels)
        if len(labs) <= 1:
            return labels
        share = {
            int(lab): sum(wins[i][1] - wins[i][0] for i in np.flatnonzero(labels == lab)) / total
            for lab in labs
        }
        small = [
            lab
            for lab in labs
            if share[int(lab)] < min_share or (labels == lab).sum() < min_windows
        ]
        if not small:
            return labels
        victim = min(small, key=lambda lab: share[int(lab)])
        cents = centroids(embs, labels)
        others = [lab for lab in cents if lab != victim]
        target = max(others, key=lambda lab: float(cents[lab] @ cents[int(victim)]))
        labels[labels == victim] = target


def turns_from_windows(
    wins: list[tuple[int, int]],
    labels: list[int] | np.ndarray,
    regions: list[tuple[int, int]],
    min_turn_ms: int = 400,
) -> list[Turn]:
    if not wins:
        return []
    order_idx = np.argsort([(s + e) / 2 for s, e in wins], kind="stable")
    centers = np.array([(wins[i][0] + wins[i][1]) / 2 for i in order_idx])
    labels = np.asarray(labels)[order_idx]
    starts = [np.arange(s, e, FRAME_MS) for s, e in regions if e > s]
    if not starts:
        return []
    frames = np.concatenate(starts)
    ends = np.concatenate(
        [np.minimum(np.arange(s, e, FRAME_MS) + FRAME_MS, e) for s, e in regions if e > s]
    )
    mids = frames + FRAME_MS / 2
    hi = np.clip(np.searchsorted(centers, mids), 0, len(centers) - 1)
    lo = np.clip(hi - 1, 0, len(centers) - 1)
    nearest = np.where(np.abs(centers[lo] - mids) <= np.abs(centers[hi] - mids), lo, hi)
    labs = labels[nearest]
    # a new run starts where the label changes or the speech has a gap
    new_run = np.ones(len(frames), dtype=bool)
    new_run[1:] = (labs[1:] != labs[:-1]) | (frames[1:] > ends[:-1])
    first = np.flatnonzero(new_run)
    last = np.append(first[1:] - 1, len(frames) - 1)
    runs: list[list[int]] = [
        [int(labs[a]), int(frames[a]), int(ends[b])] for a, b in zip(first, last, strict=True)
    ]
    # absorb short runs sandwiched between two runs of the same speaker
    changed = True
    while changed:
        changed = False
        for i in range(1, len(runs) - 1):
            lab, s, e = runs[i]
            if e - s < min_turn_ms and runs[i - 1][0] == runs[i + 1][0] != lab:
                runs[i][0] = runs[i - 1][0]
                changed = True
        merged: list[list[int]] = []
        for r in runs:
            if merged and merged[-1][0] == r[0] and r[1] - merged[-1][2] < 1000:
                merged[-1][2] = r[2]
            else:
                merged.append(list(r))
        runs = merged
    order: dict[int, str] = {}
    for lab, _s, _e in runs:
        if lab not in order:
            order[lab] = f"S{len(order) + 1}"
    return [Turn(speaker=order[lab], start_ms=int(s), end_ms=int(e)) for lab, s, e in runs]


def purity(unit_start: int, unit_end: int, speaker: str, turns: list[Turn]) -> float:
    """Share of the unit's speech (as covered by turns) that belongs to its own speaker."""
    own = other = 0
    for t in turns:
        ov = min(unit_end, t.end_ms) - max(unit_start, t.start_ms)
        if ov > 0:
            if t.speaker == speaker:
                own += ov
            else:
                other += ov
    return 1.0 if own + other == 0 else own / (own + other)
