"""Timing fit: make each dubbed line fit the time slot of the original line.

The timing ladder (quality first; changing speed is the last resort):
  1. Pick the translation variant {short, target, long} whose PREDICTED length fits the
     slot (costs no GPU).
  2. Borrow silence on both sides, following the "bounded ripple" rule.
  3. Engines with duration control ("exact") are squeezed only within safe_ratio
     [0.92, 1.08].
  4. (final profile) Ask the LLM for a shorter rewrite, at most 2 times.
  5. Time-stretch the audio. Total speed-up target <= 1.10, hard cap 1.20; beyond that the
     unit is flagged.

Bounded ripple: the silence between two lines is split at its midpoint (minus a 120 ms
minimum pause). A line may only borrow the UNUSED part of its neighbour's half, and that
part is computed from the neighbour's natural length and slot, not from where the
neighbour ended up. So the placement of line i depends only on i-1, i and i+1: editing one
line never re-flows the whole video.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

from dub.contracts.v1 import Translation
from dub.core.config import TimingCfg

_VOWEL_GROUPS = re.compile(r"[aeiouy]+")
_WORD = re.compile(r"\w+", re.UNICODE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
_THAI_BASE = re.compile(r"[ก-ะาำเ-ๆ]")


def _strip_marks(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _syllables(word: str) -> int:
    if word.isdigit():
        return max(1, len(word))  # rough: about one syllable per digit when read aloud
    cjk = len(_CJK.findall(word))
    if cjk:
        return cjk
    plain = _strip_marks(word).lower()
    n = len(_VOWEL_GROUPS.findall(plain))
    if n > 1 and plain.endswith("e") and not plain.endswith(("le", "ee")):
        n -= 1  # English silent e: "cake", "time"
    return max(1, n)


def count_units(text: str, unit: str) -> int:
    """Count "speech units" to estimate how long a line takes to say.

    syllable: syllables (vi: one per word; en/es/...: vowel groups)
    char: characters (zh/ja/ko/th)
    word: words
    """
    if unit == "word":
        return max(1, len(_WORD.findall(text)))
    if unit == "char":
        n = len(_CJK.findall(text)) + len(_THAI_BASE.findall(text))
        latin = [w for w in _WORD.findall(text) if not _CJK.search(w) and not _THAI_BASE.search(w)]
        n += sum(_syllables(w) for w in latin)
        return max(1, n)
    return max(1, sum(_syllables(w) for w in _WORD.findall(text)))


def predict_ms(
    text: str, unit: str, units_per_s: float, base_ms: int = 0, per_unit_ms: int = 0
) -> int:
    """Natural length of a line: its units at the average rate, but never under the floor
    of a short line (base_ms + units * per_unit_ms; see Langpack.short_base_ms)."""
    n = count_units(text, unit)
    linear = round(n / units_per_s * 1000)
    return max(linear, base_ms + n * per_unit_ms) if (base_ms or per_unit_ms) else linear


# ---------------------------------------------------------------- step 1: pick a variant


def pick_variant(
    tr: Translation, slot_ms: int, predict: Callable[[str], int], window: float
) -> tuple[str, str, int]:
    """Return (variant name, text, predicted length in ms)."""
    candidates = (("target", tr.target), ("short", tr.short), ("long", tr.long))
    options = [(n, t, predict(t)) for n, t in candidates if t and t.strip()]
    if not options:
        raise ValueError(f"Empty translation for {tr.unit_id}")
    slot = max(slot_ms, 1)
    for name, text, pred in options:  # 'target' is checked first
        if abs(pred / slot - 1.0) <= window:
            return name, text, pred
    fitting = [o for o in options if o[2] <= slot * (1 + window)]
    if fitting:
        return max(fitting, key=lambda o: o[2])  # longest that still fits: natural, less silence
    return min(options, key=lambda o: o[2])  # everything is too long: take the shortest


def exact_target_ms(slot_ms: int, natural_ms: int, safe: tuple[float, float]) -> int:
    """Length to request from an "exact" engine (OmniVoice): as close to the slot as
    possible while staying inside safe_ratio."""
    lo, hi = natural_ms / safe[1], natural_ms / safe[0]
    return round(min(max(slot_ms, lo), hi))


# ---------------------------------------------------------------- steps 2 + 5: place and stretch


@dataclass
class Neighbor:
    natural_ms: int  # predicted natural length of the neighbouring line
    slot_ms: int
    gap_ms: int  # silence between this line and the neighbour
    same_speaker: bool


@dataclass
class Fit:
    start_ms: int
    end_ms: int
    stretch: float  # atempo factor applied to the clip (>1 = faster)
    rate: float  # total factor: natural / final length
    room_before_ms: int
    room_after_ms: int
    flags: list[str] = field(default_factory=list)


def _share(gap: int, t: TimingCfg) -> float:
    return max(0.0, (gap - t.min_pause_ms) / 2)


def _need(nb: Neighbor, t: TimingCfg) -> float:
    return max(0.0, nb.natural_ms / t.rate_cap - nb.slot_ms)


def fit_unit(
    *,
    slot_start: int,
    slot_end: int,
    natural_ms: int,
    clip_ms: int,
    prev: Neighbor | None,
    next_: Neighbor | None,
    t: TimingCfg,
    media_end_ms: int | None = None,
) -> Fit:
    slot = slot_end - slot_start
    k = 0.5 if t.strict else 1.0
    early_cap, late_cap = t.early_start_ms * k, t.late_end_ms * k

    if prev is not None:
        own = _share(prev.gap_ms, t)
        unused = max(0.0, own - _need(prev, t) / 2)
        room_before = min(early_cap, own + (0.0 if prev.same_speaker else unused))
    else:
        room_before = min(early_cap, float(slot_start))

    if next_ is not None:
        own = _share(next_.gap_ms, t)
        if next_.same_speaker:
            room_after = min(late_cap, own)
        else:
            unused = max(0.0, own - _need(next_, t) / 2)
            room_after = min(late_cap, max(0.0, next_.gap_ms - t.tail_guard_ms), own + unused)
    else:
        room_after = late_cap
        if media_end_ms is not None:
            room_after = max(0.0, min(room_after, media_end_ms - slot_end))

    flags: list[str] = []
    if clip_ms <= slot:
        start, final_len, stretch = slot_start, clip_ms, 1.0
    else:
        extra = clip_ms - slot
        after = min(extra, room_after)  # ending late is less noticeable than starting early
        before = min(extra - after, room_before)
        avail = slot + after + before
        start = round(slot_start - before)
        if clip_ms <= avail:
            final_len, stretch = clip_ms, 1.0
        else:
            allowed = natural_ms / t.rate_cap  # shortest length we accept
            final_len = min(max(avail, allowed), clip_ms)
            stretch = clip_ms / final_len
            if final_len > avail + 1:
                flags.append("overflow")
    rate = natural_ms / max(final_len, 1)
    if rate > t.rate_p95 + 1e-9:
        flags.append("fast")
    return Fit(
        start_ms=int(start),
        end_ms=int(round(start + final_len)),
        stretch=round(stretch, 4),
        rate=round(rate, 4),
        room_before_ms=int(room_before),
        room_after_ms=int(room_after),
        flags=flags,
    )
