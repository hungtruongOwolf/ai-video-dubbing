"""Risk score per unit: which lines a human should listen to first.

M2 uses hand-set weights over the signals the pipeline already has. M4 adds the QA gate
(ASR back-check, speaker similarity, truncation) and later learns the weights from your own
review decisions.
"""

from __future__ import annotations

from dub.contracts.v1 import Placement, Translation, Unit


def score(unit: Unit, placement: Placement | None, tr: Translation | None,
          edited: bool = False) -> tuple[float, list[str]]:  # fmt: skip
    s, why = 0.0, []
    if placement:
        r = placement.rate
        if r > 1.08:
            s += (r - 1.0) * 6
            why.append(f"sped up x{r:.2f}")
        elif r < 0.75:
            s += (1.0 - r) * 2
            why.append(f"much shorter than the original (x{r:.2f})")
        if "overflow" in placement.flags:
            s += 3
            why.append("does not fit its slot")
        if "untranslated" in placement.flags:
            s += 5
            why.append("not translated")
        if placement.rewrites:
            s += 0.5 * placement.rewrites
            why.append(f"rewritten {placement.rewrites}x to fit")
    if unit.asr_conf < 0.75:
        s += (0.75 - unit.asr_conf) * 6
        why.append(f"source transcript uncertain ({unit.asr_conf:.2f})")
    if unit.slot_ms < 1300:
        s += 0.5
        why.append("very short line")
    if tr and tr.note and tr.note not in ("edited", "fake translation"):
        s += 1
        why.append(f"translator note: {tr.note}")
    if edited:
        s *= 0.3  # already looked at by a human
    return round(s, 2), why
