"""Forms of address: check every translated line against the cast sheet, by rules.

The translator returns, per line, who it is spoken to (`listener`) and the forms of address it
used (`self_term`, `other_term`). The cast sheet holds the expected pair for each speaker ->
listener relation in the target language (e.g. vi: grandson -> grandfather = "cháu" / "ông").
A line is flagged when:
- its declared terms differ from the expected ones;
- it uses a generic pronoun ("tôi", "bạn", "cậu", "mày"...) where the relation calls for
  kinship terms (the langpack lists the generic words and the address words of a language);
- (whole-script pass) its terms differ from what the same speaker uses for the same listener
  in most other lines, while no change of relation (`since_ms`) explains it.

Flagged lines are translated again with the problem stated (final profile, up to 2 rounds),
and what remains is shown at the top of the review page. Nothing here needs a model.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from dub.contracts.v1 import AddressTerms, CastSheet, Translation, Unit


def _norm(term: str | None) -> str:
    return unicodedata.normalize("NFC", (term or "").strip().lower())


def tokens(text: str) -> list[str]:
    return [_norm(t) for t in re.findall(r"\w+", unicodedata.normalize("NFC", text))]


def expected(
    cast: CastSheet, speaker: str, listener: str | None, lang: str, at_ms: int
) -> AddressTerms | None:
    """The address terms of the latest relation speaker -> listener valid at `at_ms`."""
    if not listener:
        return None
    rels = [
        r
        for r in cast.relations
        if r.speaker == speaker and r.listener == listener and r.since_ms <= at_ms
    ]
    if not rels:
        return None
    rel = max(rels, key=lambda r: r.since_ms)
    return rel.address.get(lang)


def check_line(
    unit: Unit,
    text: str,
    tr: Translation,
    cast: CastSheet,
    lang: str,
    generic: set[str],
) -> list[str]:
    exp = expected(cast, unit.speaker, tr.listener, lang, unit.start_ms)
    if exp is None:
        return []
    issues = []
    for role, used, want in (
        ("self", tr.self_term, exp.self_term),
        ("other", tr.other_term, exp.other_term),
    ):
        if used and want and _norm(used) != _norm(want):
            issues.append(f"{role} term '{used}' but the cast sheet says '{want}'")
    wanted = {_norm(exp.self_term), _norm(exp.other_term)}
    stray = sorted({t for t in tokens(text) if t in generic and t not in wanted})
    if stray and (wanted - generic - {""}):
        issues.append(f"generic pronoun {', '.join(repr(t) for t in stray)} in a kinship relation")
    return issues


def consistency(
    units: list[Unit], translations: dict[str, Translation], cast: CastSheet, min_lines: int = 3
) -> dict[str, str]:
    """Lines whose (self, other) terms differ from the majority for the same speaker ->
    listener pair, within one stretch of the relation (between two `since_ms` changes)."""

    def epoch(u: Unit, listener: str) -> int:
        starts = [
            r.since_ms
            for r in cast.relations
            if r.speaker == u.speaker and r.listener == listener and r.since_ms <= u.start_ms
        ]
        return max(starts, default=0)

    groups: dict[tuple, list[tuple[str, tuple[str, str]]]] = {}
    for u in units:
        tr = translations.get(u.id)
        if not tr or not tr.listener or not (tr.self_term or tr.other_term):
            continue
        key = (u.speaker, tr.listener, epoch(u, tr.listener))
        groups.setdefault(key, []).append((u.id, (_norm(tr.self_term), _norm(tr.other_term))))
    out: dict[str, str] = {}
    for (spk, lst, _), rows in groups.items():
        if len(rows) < min_lines:
            continue
        (top, n), *_ = Counter(t for _, t in rows).most_common(1)
        if n <= len(rows) / 2:
            continue  # no clear majority: nothing to be consistent with
        for uid, terms in rows:
            if terms != top and not _compatible(terms, top):
                out[uid] = (
                    f"{spk} -> {lst} uses '{terms[0] or '-'}/{terms[1] or '-'}' here but "
                    f"'{top[0] or '-'}/{top[1] or '-'}' in {n} of {len(rows)} lines"
                )
    return out


def _compatible(a: tuple[str, str], b: tuple[str, str]) -> bool:
    """A line may drop one of the two terms (Vietnamese often omits the pronoun)."""
    return all(x == y or not x or not y for x, y in zip(a, b, strict=True))
