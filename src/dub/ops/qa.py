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
import unicodedata
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
    foreign: bool = False  # heard as the source language (an accent heard as foreign)

    def mark_foreign(self, lang: str) -> None:
        self.foreign = True
        self.passed = False
        self.reasons.append(f"accent: heard as {lang}")

    @property
    def score(self) -> float:
        """Higher is better: used to pick among takes when none passes."""
        sim = 1.0 if self.sim_ratio is None else min(self.sim_ratio, 1.2)
        return -2.0 * self.cer + 0.5 * sim + 0.5 * self.tail - (0.5 if self.foreign else 0.0)

    def as_dict(self) -> dict:
        return {
            "cer": round(self.cer, 3),
            "sim_ratio": None if self.sim_ratio is None else round(self.sim_ratio, 3),
            "tail": round(self.tail, 3),
            "passed": self.passed,
            "reasons": self.reasons,
            "closest": self.closest,
        }


def _bare(word: str) -> str:
    return unicodedata.normalize("NFC", "".join(ch for ch in word if ch.isalnum())).casefold()


def fold_particles(text: str, heard: str, groups: list[list[str]] | None) -> str:
    """`heard` with its last word replaced by the line's last word when both are final
    particles of one group (langpack `final_particles`): "nó ăn được hả" for "Nó ăn được à?"
    is the line heard right. Two of the drama's three short lines that failed QA differed only
    there (à/hả, ư/à); a 4-syllable line has no room for one wrong letter (CER limit 0.15)."""
    if not groups:
        return heard
    tw, hw = text.split(), heard.split()
    if not tw or not hw:
        return heard
    a, b = _bare(tw[-1]), _bare(hw[-1])
    sets = [{_bare(p) for p in g} for g in groups]
    if a != b and any(a in g and b in g for g in sets):
        return " ".join([*hw[:-1], tw[-1]])
    return heard


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


# Fastest speech a clip can hold, per speech unit: syllables (vi), characters (zh, ja), words.
MAX_UNITS_PER_S = {"syllable": 8.0, "char": 10.0, "word": 5.0}


def plausible(words: list[Word], dur_ms: int, expected: str, unit: str, joiner: str) -> list[Word]:
    """The words a back-check can really have heard in a clip this long.

    Whisper models invent text on very short clips: PhoWhisper heard a 600 ms "Gầy rồi." as
    "cay rồi. hàng trăm triệu tập theo khuynh hướng vô sản trong..." (M8). Nobody says that in
    600 ms. When the transcript holds more speech units than the clip can (MAX_UNITS_PER_S,
    plus one), the run of consecutive words that fits and is closest to the expected text is
    kept; the rest was never in the audio.
    """
    from dub.ops.timing import count_units

    cap = dur_ms / 1000 * MAX_UNITS_PER_S.get(unit, 8.0) + 1
    if not words or count_units(join_words(words, joiner), unit) <= cap:
        return words
    best, best_err = words[:0], 2.0
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            if count_units(join_words(words[i:j], joiner), unit) > cap:
                break
            err = cer(expected, join_words(words[i:j], joiner))
            if err < best_err:
                best, best_err = words[i:j], err
    return list(best)


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
    particles: list[list[str]] | None = None,
) -> QaResult:
    heard = fold_particles(text, join_words(words, joiner), particles)
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
