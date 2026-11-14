"""QA gate for one dubbed take: is it the right words, in the right voice, said to the end?

Three checks per take (the design's "4 checks" minus duration, which the timing fit owns):
- words: CER between the line and what the back-check ASR heard;
- voice: the take must sound more like its own speaker than like anyone else in the video
  (cosine to each speaker's centroid), and its similarity to its own centroid, divided by how
  similar the speaker's original lines are to it (`self_sim`), must not collapse. Measured on
  the drama (E10): cross-lingual clones sit around 0.75 of the speaker's own similarity, so the
  floor is low (0.5); the "closer to someone else" test is what catches a wrong voice. Clips
  under 1.2 s are not judged on voice (their embeddings are unreliable);
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
    sim_ratio_min: float = 0.5
    tail_min: float = 0.6
    other_margin: float = 0.05  # fail if another speaker's centroid is closer by this much
    voice_min_ms: int = 1200
    # Another speaker whose centroid is this close to the line's speaker is a twin voice (two
    # women of a drama the attribution told apart by the words): no "sounds like" check there.
    twin_sim: float = 0.8


@dataclass
class QaResult:
    cer: float
    sim_ratio: float | None  # None when no reference voice was available
    tail: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    closest: str | None = None  # the speaker whose centroid is closest to the take

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
            "closest": self.closest,
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
    speaker: str | None = None,
    centroids: dict[str, list[float]] | None = None,
    dur_ms: int | None = None,
) -> QaResult:
    heard = join_words(words, joiner)
    c = cer(text, heard)
    tail = tail_heard(text, heard)
    ratio = None
    closest = None
    reasons = []
    judge_voice = take_vec and centroid and self_sim and (dur_ms or 10**9) >= limits.voice_min_ms
    if judge_voice:
        own = cosine(take_vec, centroid)
        ratio = own / self_sim
        if centroids and speaker:
            sims = {
                s: cosine(take_vec, v)
                for s, v in centroids.items()
                if v and (s == speaker or cosine(v, centroid) < limits.twin_sim)
            }
            closest = max(sims, key=sims.get) if sims else None
            if closest and closest != speaker and sims[closest] > own + limits.other_margin:
                reasons.append(f"voice: sounds more like {closest} than {speaker}")
    if c > limits.cer_max:
        reasons.append(f"words: CER {c:.0%}")
    if ratio is not None and ratio < limits.sim_ratio_min:
        reasons.append(f"voice: similarity {ratio:.2f} of the speaker's own")
    if tail < limits.tail_min:
        reasons.append(f"ending: only {tail:.0%} of the last words heard")
    return QaResult(
        cer=c, sim_ratio=ratio, tail=tail, passed=not reasons, reasons=reasons, closest=closest
    )
