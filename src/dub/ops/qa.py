"""QA gate for one dubbed take: is it the right words, in the right voice, said to the end?

Three checks per take (the design's "4 checks" minus duration, which the timing fit owns):
- words: CER between the line and what the back-check ASR heard;
- voice: cosine similarity of the take's speaker embedding to the speaker's centroid, divided
  by how similar the speaker's own original lines are to that centroid (`self_sim`). A ratio
  of 1.0 means "as close as the speaker is to themself";
- ending: the share of the last quarter of the line that the ASR heard. A take that stops
  before its last words ("truncated") fails here even when the CER is still small.

A take passes when all three are within the `qa:` limits of dub.yaml. When none of the takes
passes, the best one is kept and the line is flagged for review.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from dub.contracts.v1 import Word, join_words
from dub.ops.align import _norm_chars, cer


@dataclass
class QaLimits:
    cer_max: float = 0.15
    sim_ratio_min: float = 0.75
    tail_min: float = 0.6


@dataclass
class QaResult:
    cer: float
    sim_ratio: float | None  # None when no reference voice was available
    tail: float
    passed: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Higher is better: used to pick among takes when none passes."""
        sim = 1.0 if self.sim_ratio is None else min(self.sim_ratio, 1.2)
        return -2.0 * self.cer + 0.5 * sim + 0.5 * self.tail

    def as_dict(self) -> dict:
        return {
            "cer": round(self.cer, 3),
            "sim_ratio": None if self.sim_ratio is None else round(self.sim_ratio, 3),
            "tail": round(self.tail, 3),
            "passed": self.passed,
            "reasons": self.reasons,
        }


def tail_heard(text: str, heard: str, share: float = 0.25) -> float:
    """Fraction of the last `share` of the line's letters that the ASR also produced."""
    ref = [c for _, c in _norm_chars(text, strip_marks=True)]
    hyp = [c for _, c in _norm_chars(heard, strip_marks=True)]
    if not ref:
        return 1.0
    start = int(len(ref) * (1 - share))
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    matched = set()
    for a, _b, size in sm.get_matching_blocks():
        matched.update(range(a, a + size))
    tail = range(start, len(ref))
    return sum(1 for i in tail if i in matched) / max(len(tail), 1)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def judge(
    text: str,
    words: list[Word],
    joiner: str,
    take_vec: list[float] | None,
    centroid: list[float] | None,
    self_sim: float | None,
    limits: QaLimits,
) -> QaResult:
    heard = join_words(words, joiner)
    c = cer(text, heard)
    tail = tail_heard(text, heard)
    ratio = None
    if take_vec and centroid and self_sim:
        ratio = cosine(take_vec, centroid) / self_sim
    reasons = []
    if c > limits.cer_max:
        reasons.append(f"words: CER {c:.0%}")
    if ratio is not None and ratio < limits.sim_ratio_min:
        reasons.append(f"voice: similarity {ratio:.2f} of the speaker's own")
    if tail < limits.tail_min:
        reasons.append(f"ending: only {tail:.0%} of the last words heard")
    return QaResult(cer=c, sim_ratio=ratio, tail=tail, passed=not reasons, reasons=reasons)
