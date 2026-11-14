"""Apply human edits to the pipeline's units. Pure functions: same inputs, same outputs.

Where each kind of edit enters the pipeline decides what it re-runs:

    split, merge_next   before translation: the translation window that holds the unit is
                        translated again (other windows keep their cache)
    text                replaces that unit's translation; nothing else is translated again
    speaker, nudge,     after translation: they change the voice or the time slot, so only
    seed                that unit's TTS and the fit of its neighbours re-run
    lock                keeps the line away from automatic rewrites
    undo                cancels an earlier edit

Unit ids stay deterministic: a split turns `u0012345-S1` into `u0012345-S1.a` and
`u0012345-S1.b`, and a merge gives `<first>+<second>`. Each derived unit lists its
`parent_ids`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dub.contracts.v1 import (
    AddressTerms,
    CastSpeaker,
    ContextSheet,
    Edit,
    Relation,
    Translation,
    Unit,
    Word,
    join_words,
)

STRUCTURE = {"split", "merge_next"}


def effective(edits: list[Edit]) -> list[Edit]:
    """Edits in order, without the ones an `undo` cancelled (and without the undos)."""
    ordered = sorted(edits, key=lambda e: e.id)
    undone = {e.value for e in ordered if e.field == "undo"}
    return [e for e in ordered if e.field != "undo" and e.id not in undone]


def _join(words: list[Word], joiner: str) -> str:
    return join_words(words, joiner)


def _words_in(unit: Unit, words: list[Word]) -> list[Word]:
    return [w for w in words if unit.start_ms <= w.start_ms < unit.end_ms + 1
            and w.end_ms <= unit.end_ms + 1]  # fmt: skip


def apply_structure(
    units: list[Unit], words: list[Word], edits: list[Edit], joiner: str = " "
) -> tuple[list[Unit], list[str]]:
    """Split/merge edits in order. Returns (units, warnings for edits that no longer apply)."""
    out = list(units)
    warnings: list[str] = []
    for e in edits:
        if e.field not in STRUCTURE:
            continue
        idx = next((i for i, u in enumerate(out) if u.id == e.unit_id), None)
        if idx is None:
            warnings.append(f"{e.field} {e.unit_id}: unit no longer exists")
            continue
        u = out[idx]
        if e.field == "split":
            ws = _words_in(u, words)
            if len(ws) < 2:
                warnings.append(f"split {u.id}: fewer than two words")
                continue
            at = (e.value or {}).get("at_ms") if isinstance(e.value, dict) else None
            at = at if at is not None else (u.start_ms + u.end_ms) // 2
            # cut between the two words whose gap is nearest to at_ms
            k = min(
                range(1, len(ws)), key=lambda i: abs((ws[i - 1].end_ms + ws[i].start_ms) / 2 - at)
            )
            left, right = ws[:k], ws[k:]
            parents = u.parent_ids or [u.id]
            out[idx : idx + 1] = [
                u.model_copy(update={"id": f"{u.id}.a", "start_ms": left[0].start_ms,
                                     "end_ms": left[-1].end_ms, "text": _join(left, joiner),
                                     "n_words": len(left), "parent_ids": parents}),
                u.model_copy(update={"id": f"{u.id}.b", "start_ms": right[0].start_ms,
                                     "end_ms": right[-1].end_ms, "text": _join(right, joiner),
                                     "n_words": len(right), "parent_ids": parents}),
            ]  # fmt: skip
        else:  # merge_next
            if idx + 1 >= len(out):
                warnings.append(f"merge_next {u.id}: no next unit")
                continue
            v = out[idx + 1]
            if v.speaker != u.speaker:
                warnings.append(f"merge_next {u.id}: next unit has another speaker")
                continue
            sep = joiner if joiner else ""
            out[idx : idx + 2] = [u.model_copy(update={
                "id": f"{u.id}+{v.id}", "end_ms": v.end_ms, "text": f"{u.text}{sep}{v.text}",
                "n_words": u.n_words + v.n_words,
                "sentence_ids": u.sentence_ids + [s for s in v.sentence_ids
                                                  if s not in u.sentence_ids],
                "asr_conf": min(u.asr_conf, v.asr_conf),
                "parent_ids": (u.parent_ids or [u.id]) + (v.parent_ids or [v.id]),
            })]  # fmt: skip
    return out, warnings


@dataclass
class UnitEdits:
    """The per-unit (non-structural) edits, latest value per unit and field."""

    text: dict[str, str] = field(default_factory=dict)
    seed: dict[str, int] = field(default_factory=dict)
    speaker: dict[str, str] = field(default_factory=dict)
    nudge: dict[str, tuple[int, int]] = field(default_factory=dict)
    locked: set[str] = field(default_factory=set)

    @classmethod
    def from_edits(cls, edits: list[Edit]) -> UnitEdits:
        ue = cls()
        for e in edits:
            if e.field == "text":
                ue.text[e.unit_id] = str(e.value)
            elif e.field == "seed":
                ue.seed[e.unit_id] = int(e.value)
            elif e.field == "speaker":
                ue.speaker[e.unit_id] = str(e.value)
            elif e.field == "nudge":
                v = e.value or {}
                ue.nudge[e.unit_id] = (int(v.get("start_ms", 0)), int(v.get("end_ms", 0)))
            elif e.field == "lock":
                (ue.locked.add if e.value else ue.locked.discard)(e.unit_id)
        return ue

    def is_protected(self, unit_id: str) -> bool:
        return unit_id in self.text or unit_id in self.locked

    def apply_to_units(self, units: list[Unit]) -> list[Unit]:
        """Speaker and slot changes. Slots never overlap a neighbour or go below 200 ms."""
        out = []
        for i, u in enumerate(units):
            upd: dict = {}
            if u.id in self.speaker:
                upd["speaker"] = self.speaker[u.id]
            if u.id in self.nudge:
                ds, de = self.nudge[u.id]
                lo = units[i - 1].end_ms if i > 0 else 0
                hi = units[i + 1].start_ms if i + 1 < len(units) else u.end_ms + max(de, 0)
                start = max(lo, u.start_ms + ds)
                end = min(hi, u.end_ms + de)
                if end - start >= 200:
                    upd["start_ms"], upd["end_ms"] = start, end
            out.append(u.model_copy(update=upd) if upd else u)
        return out

    def apply_to_translation(self, tr: Translation) -> Translation:
        t = self.text.get(tr.unit_id)
        if t is None:
            return tr
        return tr.model_copy(update={"short": t, "target": t, "long": t, "note": "edited"})


SPEAKER_FIELDS = ("name", "gender", "age_band", "role", "style")


def apply_cast(sheet: ContextSheet, edits: list[Edit]) -> ContextSheet:
    """Human corrections of the cast sheet, applied over the LLM's (later edits win).

    A speaker correction locks that speaker; a relation correction sets the address terms of
    one speaker -> listener pair in one language. Only the translation windows where those
    speakers appear are translated again (their cast rows are part of the window's key).
    """
    cast = sheet.cast.model_copy(deep=True)
    for e in edits:
        if e.field != "cast" or not isinstance(e.value, dict):
            continue
        v = e.value
        spk = str(v.get("speaker", ""))
        if not spk:
            continue
        if "listener" in v:
            lang = str(v.get("lang", "vi"))
            since = int(v.get("since_ms", 0))
            rel = next((r for r in cast.relations if r.speaker == spk
                        and r.listener == v["listener"] and r.since_ms == since), None)  # fmt: skip
            if rel is None:
                rel = Relation(speaker=spk, listener=str(v["listener"]), since_ms=since)
                cast.relations.append(rel)
            terms = rel.address.get(lang, AddressTerms())
            rel.address[lang] = terms.model_copy(update={
                k: v[k] for k in ("self_term", "other_term") if k in v
            })  # fmt: skip
            rel.conf = 1.0
            rel.evidence = ["confirmed by the user"]
            continue
        row = next((s_ for s_ in cast.speakers if s_.id == spk), None)
        if row is None:
            row = CastSpeaker(id=spk)
            cast.speakers.append(row)
        for k in SPEAKER_FIELDS:
            if k in v:
                setattr(row, k, v[k])
        row.locked = True
        row.conf = 1.0
        row.evidence = ["confirmed by the user"]
    return sheet.model_copy(update={"cast": cast})
