"""Risk score per unit: which lines a human should listen to first.

Hand-set weights over the signals the pipeline already has. Since M3 that includes the
back-check CER: the dubbed clip is transcribed again and compared with the line. M4 adds
speaker similarity and truncation checks, and later learns the weights from your own review
decisions.
"""

from __future__ import annotations

from dub.contracts.v1 import Placement, Translation, Unit


def score(unit: Unit, placement: Placement | None, tr: Translation | None,
          edited: bool = False, cer: float | None = None, qa: dict | None = None,
          purity: float | None = None, address: str | None = None,
          speaker_unsure: bool = False,
          ) -> tuple[float, list[str]]:  # fmt: skip
    s, why = 0.0, []
    if placement and placement.variant == "original":
        # a song, a cry, a laugh kept as it is: nothing dubbed to listen to (M10: two baby cries
        # topped the drama's list)
        return 0.0, ["kept in the original audio"]
    if qa and qa.get("passed") is False:
        # every take failed the gate: the one kept is the best of bad ones ("căng đến đau"
        # heard as "căn đến đâu" scored 1.96 and was not listed)
        s += 3
        why.append("failed the QA gate: " + "; ".join(qa.get("reasons", [])[:2]))
    if speaker_unsure:
        s += 2
        why.append("who says it is a guess (the script does not show it)")
    if address:
        s += 3
        why.append(f"forms of address: {address}")
    if cer is not None and cer > 0.15:
        s += min(cer, 1.0) * 8
        why.append(f"the dub does not sound like the text (CER {cer:.0%})")
    if qa and any(r.startswith("voice") for r in qa.get("reasons", [])):
        s += 4
        why.append(next(r for r in qa["reasons"] if r.startswith("voice")))
    if qa and qa.get("tail", 1.0) < 0.6:
        s += 3
        why.append("the end of the line may be cut off")
    if purity is not None and purity < 0.8:
        s += (0.8 - purity) * 10
        why.append(f"another voice in the original line ({1 - purity:.0%})")
    if placement:
        r = placement.rate
        if r > 1.08:
            s += (r - 1.0) * 6
            why.append(f"sped up x{r:.2f}")
        elif r < 0.75:
            s += (1.0 - r) * 2
            why.append(f"drawn out to fill its slot (x{r:.2f})")
        slot = placement.slot_end_ms - placement.slot_start_ms
        fill = (placement.end_ms - placement.start_ms) / max(1, slot)
        if slot >= 1200 and fill < 0.6:
            # the voice stops long before the mouth does (VieNeu lines have no length control)
            s += (0.6 - fill) * 6
            why.append(f"fills {fill:.0%} of its slot")
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
