"""Sentence-boundary F1: how well the pipeline's sentences match a reference segmentation.

The two sides may not have the same text (the reference is what was really said, ours is
what the ASR heard), so both are reduced to letters and digits (no case, no diacritics) and
aligned character by character. Every reference sentence end is carried over to our side
through that alignment. A boundary counts as found if ours is within `tol` words of it
(design metric: +-1 word). The end of the text is not a boundary.

Words are whitespace tokens; dictionary words for Thai (PyThaiNLP); single characters for
Chinese and Japanese.

Not every extra boundary is an error for dubbing: an ASR often puts a full stop where the
reference has a comma ("...all three. For we meet..."). `soft_fp` counts our extra boundaries
that fall where the reference has clause punctuation (, ; : - and their CJK forms), `hard_fp`
the others (a cut inside a phrase). `clause_p` = (hits + soft_fp) / ours.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from dub.ops.align import _norm_chars
from dub.ops.script import base_lang, is_no_space


@dataclass
class F1:
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gold: int
    hits: int = 0
    soft_fp: int | None = None  # extra boundaries at reference clause punctuation
    hard_fp: int | None = None  # extra boundaries inside a phrase
    unmatched: list[int] = field(default_factory=list)  # our boundaries with no reference match

    def as_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "p": round(self.precision, 4),
            "r": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "n_pred": self.n_pred,
            "n_gold": self.n_gold,
        }
        if self.soft_fp is not None:
            out["soft_fp"] = self.soft_fp
            out["hard_fp"] = self.hard_fp or 0
            ok = self.hits + self.soft_fp
            out["clause_p"] = round(ok / self.n_pred, 4) if self.n_pred else 1.0
        return out


CLAUSE_PUNCT = set(",;:-–—，、；：")


def match_f1(pred: list[int], gold: list[int], tol: int = 1) -> F1:
    """One-to-one matching of boundary positions within `tol`, greedy by distance."""
    pairs = sorted(
        (abs(p - g), i, j)
        for i, p in enumerate(pred)
        for j, g in enumerate(gold)
        if abs(p - g) <= tol
    )
    used_p, used_g, hits = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            hits += 1
    prec = hits / len(pred) if pred else (1.0 if not gold else 0.0)
    rec = hits / len(gold) if gold else 1.0
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    unmatched = [p for i, p in enumerate(pred) if i not in used_p]
    return F1(prec, rec, f, len(pred), len(gold), hits, unmatched=unmatched)


def micro(scores: list[F1]) -> F1:
    """Pool many documents: total hits over total boundaries."""
    hits = sum(s.hits for s in scores)
    n_pred = sum(s.n_pred for s in scores)
    n_gold = sum(s.n_gold for s in scores)
    prec = hits / n_pred if n_pred else 1.0
    rec = hits / n_gold if n_gold else 1.0
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return F1(prec, rec, f, n_pred, n_gold, hits)


def _words(s: str, lang: str) -> list[str]:
    if base_lang(lang) == "th":
        from pythainlp.tokenize import word_tokenize

        return [w for w in word_tokenize(s, engine="newmm") if not w.isspace()]
    return list(s.replace(" ", "")) if is_no_space(lang) else s.split()


def _stream(sentences: list[str], lang: str) -> tuple[list[str], list[int], list[int]]:
    """Normalized chars, the word index of each char, and the word index ending each sentence."""
    chars: list[str] = []
    word_of: list[int] = []
    ends: list[int] = []
    w = -1
    for s in sentences:
        tokens = _words(s, lang)
        for tok in tokens:
            norm = [c for _, c in _norm_chars(tok, strip_marks=True)]
            if not norm:
                continue
            w += 1
            chars.extend(norm)
            word_of.extend([w] * len(norm))
        if w >= 0 and (not ends or ends[-1] != w):
            ends.append(w)
    return chars, word_of, ends


def boundary_f1(pred_sentences: list[str], gold_sentences: list[str], lang: str, tol: int = 1):
    """F1 of our sentence ends against the reference sentence ends (in our word positions).

    A reference boundary is carried over by the last character of its sentence. When the ASR
    did not produce that character (a number read out: "832" vs "tam tram ba muoi hai"), the
    first character of the next sentence is used instead: the boundary is just before it.
    """
    pc, pw, pends = _stream(pred_sentences, lang)
    gc, _, _ = _stream(gold_sentences, lang)
    spans, n = [], 0  # (first, last) char index of each gold sentence in the gold stream
    for s in gold_sentences:
        m = len(_stream([s], lang)[0])
        if m:
            spans.append((n, n + m - 1))
        n += m
    sm = difflib.SequenceMatcher(None, gc, pc, autojunk=False)
    g2p: dict[int, int] = {}
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            g2p[a + k] = b + k
    matched = sorted(g2p)
    gold: list[int] = []
    for (_, last), (first_next, _) in zip(spans, spans[1:], strict=False):
        if last in g2p:
            gold.append(pw[g2p[last]])
        elif first_next in g2p and g2p[first_next] > 0:
            gold.append(pw[g2p[first_next]] - 1)
        else:
            before = [m for m in matched if m < last]
            if before:
                gold.append(pw[g2p[before[-1]]])
    gold = sorted(set(gold))
    res = match_f1(pends[:-1], gold, tol)
    # Classify our extra boundaries: does the reference have clause punctuation there?
    gold_punct = _clause_after(gold_sentences, lang)  # per gold char: punctuation after it
    p2g = {b: a for a, b in g2p.items()}
    last_char = {}  # our word index -> its last char in our stream
    for ci, wi in enumerate(pw):
        last_char[wi] = ci
    soft = 0
    for wi in res.unmatched:
        ci = last_char.get(wi)
        g = p2g.get(ci) if ci is not None else None
        if g is not None and gold_punct[g]:
            soft += 1
    res.soft_fp = soft
    res.hard_fp = len(res.unmatched) - soft
    return res


def _clause_after(sentences: list[str], lang: str) -> list[bool]:
    """For each normalized char of the reference: is a clause punctuation mark right after its
    word (", ; : -" and CJK forms)? Same char stream as `_stream`."""
    flags: list[bool] = []
    for s in sentences:
        tokens = _words(s, lang)
        for k, tok in enumerate(tokens):
            norm = [c for _, c in _norm_chars(tok, strip_marks=True)]
            if not norm:
                continue
            nxt = tokens[k + 1] if k + 1 < len(tokens) else ""
            punct = tok.rstrip()[-1:] in CLAUSE_PUNCT or (
                bool(nxt) and all(ch in CLAUSE_PUNCT for ch in nxt.strip())
            )
            flags.extend([False] * (len(norm) - 1) + [punct])
    return flags
