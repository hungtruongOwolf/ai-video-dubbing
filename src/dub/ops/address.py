"""Forms of address: check every translated line against the cast sheet, by rules.

The translator returns, per line, who it is spoken to (`listener`) and the forms of address it
used (`self_term`, `other_term`). The cast sheet holds the expected pair for each speaker ->
listener relation in the target language (e.g. vi: grandson -> grandfather = "cháu" / "ông").
A line is flagged when:
- its declared terms differ from the expected ones;
- it uses a generic pronoun ("tôi", "bạn", "cậu", "mày"...) where the relation calls for
  kinship terms (the langpack lists the generic words and the address words of a language);
- (whole-script pass) its terms differ from what the same speaker uses for the same listener
  in most other lines, while no change of relation (`since_ms`) explains it;
- it is a question answered right away by another speaker, but declares someone else as its
  listener (turn-taking: "Can Tom listen too?" followed by the mother's answer is addressed to
  the mother, whoever spoke before).

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


def options(term: str | None) -> set[str]:
    """The cast sheet may give alternatives and notes: "cháu / cháu yêu", "con (Linda)"."""
    term = re.sub(r"\([^)]*\)", " ", term or "")
    parts = re.split(r"\s*(?:/|,|;|hoặc|or)\s*", term)
    return {_norm(p) for p in parts if _norm(p)}


def matches(used: str | None, want: str | None) -> bool:
    """A used term matches an expected one, or extends it ("cháu yêu" for "cháu")."""
    u = _norm(re.sub(r"\([^)]*\)", " ", used or ""))
    opts = options(want)
    return not opts or not u or any(u == o or u.split()[0] == o.split()[0] for o in opts)


def _name(term: str) -> bool:
    """A proper name used to address someone ("A Duyên", "Nãi Long"): always a valid
    vocative. The fix loop had replaced "A Duyên" by "em" and lost the name."""
    t = (term or "").strip()
    return bool(t) and t[0].isupper()


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
    if exp is None and tr.listener:
        # A listener with no relation in the cast sheet is someone the speaker has no tie to
        # (the gatekeeper): "tôi" to a stranger is right. Flagging it made the husband's lines
        # translated again (M10 review).
        return []
    if exp is None:
        # Listener unknown: a speaker whose every relation uses kinship terms must still not
        # fall back to a generic pronoun ("Ông đã hứa thì tôi luôn giữ lời").
        mine = [r.address[lang] for r in cast.relations if r.speaker == unit.speaker
                and lang in r.address]  # fmt: skip
        terms = (
            set().union(*(options(a.self_term) | options(a.other_term) for a in mine))
            if mine
            else set()
        )
        stray = sorted({t for t in tokens(text) if t in generic and t not in terms})
        if mine and stray and not (terms & generic):
            return [f"generic pronoun {', '.join(repr(t) for t in stray)} in a kinship relation"]
        return []
    issues = []
    for role, used, want in (
        ("self", tr.self_term, exp.self_term),
        ("other", tr.other_term, exp.other_term),
    ):
        if used and want and not matches(used, want) and not (role == "other" and _name(used)):
            issues.append(f"{role} term '{used}' but the cast sheet says '{want}'")
    wanted = options(exp.self_term) | options(exp.other_term)
    wanted |= {w.split()[0] for w in wanted if w}
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


def answered_by(units: list[Unit], i: int, max_gap_ms: int = 2000) -> str | None:
    """The speaker who answers unit i right away, if unit i is a question."""
    u = units[i]
    if not u.text.rstrip().endswith(("?", "？")) or i + 1 >= len(units):
        return None
    nxt = units[i + 1]
    if nxt.speaker != u.speaker and nxt.start_ms - u.end_ms <= max_gap_ms:
        return nxt.speaker
    return None


def listener_issues(units: list[Unit], translations: dict[str, Translation]) -> dict[str, str]:
    out = {}
    for i, u in enumerate(units):
        tr = translations.get(u.id)
        who = answered_by(units, i)
        if tr and who and tr.listener and tr.listener != who:
            out[u.id] = (
                f"listener: this question is answered right away by {who}, so it is addressed "
                f"to {who}, not {tr.listener}; use the address terms for {u.speaker} -> {who}"
            )
    return out
