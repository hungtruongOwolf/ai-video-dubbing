"""Emotion of a line, from emotion2vec embeddings (M13).

dub.ops.prosody describes how a line is said with four numbers: pitch, pitch range, level,
voicing. Those cannot tell anger from joy at the same pitch, nor a tired line from a sad one.
emotion2vec (dub.providers.emotion2vec) places each line in a learned emotion space, and
gives the probabilities of 9 classes.

Here:
- `centred`: each speaker's line embeddings minus that speaker's mean, made unit length. What
  is left is how this line differs from the speaker's usual delivery, not who speaks or in
  which language (the dub's lines are centred on the dub's own mean);
- `choose`: a speaker's candidate reference clips scored per line, the best one kept when it
  beats the usual reference by `margin`;
- `agreement`: does the dub's emotion follow the source's, line by line.
"""

from __future__ import annotations

import math

SKIP_LABELS = ("other", "unknown")  # never a line's top emotion
MIN_LINES = 3  # a speaker with fewer lines has no mean to centre on


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def centred(vectors: dict[str, list[float]], speaker_of: dict[str, str]) -> dict[str, list[float]]:
    """Unit vectors of each line minus its speaker's mean (speakers with MIN_LINES lines or
    more; the others are left out)."""
    by: dict[str, list[str]] = {}
    for k, v in vectors.items():
        if v and k in speaker_of:
            by.setdefault(speaker_of[k], []).append(k)
    out: dict[str, list[float]] = {}
    for keys in by.values():
        if len(keys) < MIN_LINES:
            continue
        units = [_unit(vectors[k]) for k in keys]
        mean = [sum(col) / len(units) for col in zip(*units, strict=True)]
        for k, u in zip(keys, units, strict=True):
            out[k] = _unit([a - b for a, b in zip(u, mean, strict=True)])
    return out


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def choose(scores: dict[str, float], default: str, margin: float) -> str:
    """The lowest-scored candidate when it beats the usual reference (`default`) by `margin`.
    A usual reference without a score of its own counts as a typical clip (the median)."""
    if not scores:
        return default
    best = min(scores, key=lambda k: scores[k])
    if default in scores:
        base = scores[default]
    else:
        vals = sorted(scores.values())
        base = vals[len(vals) // 2]
    return best if base - scores[best] > margin else default


def top(scores: list[float], labels: list[str]) -> tuple[str, float]:
    """The most likely emotion that is a real one (not "other" or "unknown")."""
    pairs = [(p, lab) for p, lab in zip(scores, labels, strict=False) if lab not in SKIP_LABELS]
    if not pairs:
        return "", 0.0
    p, lab = max(pairs)
    return lab, p


def agreement(
    rows: list[tuple[str, str, list[float], list[float], list[float], list[float]]],
    labels: list[str],
    min_p: float = 0.5,
) -> dict[str, float]:
    """rows: (line id, speaker, source vector, dub vector, source scores, dub scores).

    - emotion_cos: mean cosine of the centred source and dub vectors of a line (each side
      centred on its own speaker mean): 0 when the dub's emotion has nothing to do with the
      source's, 1 when every line moves the same way;
    - emotion_label_match: of the lines whose source emotion is clear (a class other than
      neutral at min_p or more), the share whose dub has the same top class."""
    spk = {r[0]: r[1] for r in rows}
    src = centred({r[0]: r[2] for r in rows}, spk)
    dub = centred({r[0]: r[3] for r in rows}, spk)
    both = [k for k in src if k in dub]
    out: dict[str, float] = {}
    if both:
        out["emotion_cos"] = round(sum(cos(src[k], dub[k]) for k in both) / len(both), 3)
    clear = 0
    same = 0
    for _, _, _, _, s_sc, d_sc in rows:
        lab, p = top(s_sc, labels)
        if lab and lab != "neutral" and p >= min_p:
            clear += 1
            same += top(d_sc, labels)[0] == lab
    if clear:
        out["emotion_label_match"] = round(same / clear, 3)
    out["emotion_clear_lines"] = clear
    return out
