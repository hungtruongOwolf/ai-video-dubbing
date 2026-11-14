"""Subtitle line breaking for scripts without spaces: M0 (break anywhere) vs M3 (script-aware).

For every NTREX sentence in Thai, Chinese and Japanese we build cues the way the pipeline
does (one item per sentence, 4 s each) and look at every break: between two lines of a cue
and between two cues. A break is an error when it falls:
- Thai: inside a dictionary word (PyThaiNLP newmm), or between a letter and its tone mark or
  vowel sign (a broken character cluster);
- zh/ja: inside a BudouX phrase, or against kinsoku (a line starting with "。" "」" "ょ"...,
  or ending with "「").
Also counted: lines wider than the limit (Netflix widths: tone marks take no space, half-width
characters count half in CJK).

Note: the M3 layout itself uses newmm + BudouX boundaries, so its Thai/phrase error count is
zero by construction; the useful numbers are the M0 counts (what libass-style breaking does)
and the width violations. Run: uv run python experiments/subs_layout.py
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dub.contracts.v1 import SubItem, SubsReq  # noqa: E402
from dub.ops.script import NO_END, NO_START, RLM, ZWSP, _budoux, display_len  # noqa: E402
from dub.ops.subs import build_cues  # noqa: E402
from experiments.common import data_dir, documents, save_json  # noqa: E402

CPL = {"th": 35, "zh": 16, "ja": 16}


def m0_lines(text: str, cpl: int) -> list[str]:
    """The M0 layout: characters without spaces, wrapped by len() (as the old code did)."""
    chars = list(text.replace(" ", ""))
    lines, cur = [], ""
    for ch in chars:
        if cur and len(cur + ch) > cpl:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def m3_lines(text: str, lang: str, cpl: int) -> list[str]:
    req = SubsReq(
        lang=lang, items=[SubItem(start_ms=0, end_ms=4000, text=text)], cpl=cpl, max_lines=2
    )
    out = []
    for c in build_cues(req):
        out.extend(c.text.split("\n"))
    return out


def _clean(s: str) -> str:
    return s.replace(ZWSP, "").replace(RLM, "").replace(" ", "")


def word_spans(text: str, lang: str) -> list[tuple[int, int]]:
    """(start, end) of each word/phrase in the space-free text."""
    if lang == "th":
        from pythainlp.tokenize import word_tokenize

        pieces = [w for w in word_tokenize(text, engine="newmm") if not w.isspace()]
    else:
        pieces = [p for chunk in text.split() for p in _budoux(lang).parse(chunk)]
    spans, pos = [], 0
    for p in pieces:
        p = p.replace(" ", "")
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return spans


def errors(text: str, lines: list[str], lang: str, cpl: int) -> dict[str, int]:
    flat = _clean(text)
    assert "".join(_clean(x) for x in lines) == flat, (text, lines)
    breaks, pos = [], 0
    for line in lines[:-1]:
        pos += len(_clean(line))
        breaks.append(pos)
    inside = {i for a, b in word_spans(text, lang) for i in range(a + 1, b)}
    e = {"breaks": len(breaks), "in_word": sum(1 for b in breaks if b in inside), "width": 0}
    e["cluster"] = sum(1 for b in breaks if b < len(flat) and unicodedata.category(flat[b]) == "Mn")
    e["kinsoku"] = 0
    if lang in ("zh", "ja"):
        for a, b in zip(lines, lines[1:], strict=False):
            if _clean(b)[:1] in NO_START or _clean(a)[-1:] in NO_END:
                e["kinsoku"] += 1
    e["width"] = sum(1 for x in lines if display_len(x, lang) > cpl)
    return e


def main() -> None:
    results: dict[str, dict] = {}
    for lang, cpl in CPL.items():
        sents = [s for doc in documents(lang) for s in doc]
        tot = {"M0": {}, "M3": {}}
        for s in sents:
            for name, lines in (("M0", m0_lines(s, cpl)), ("M3", m3_lines(s, lang, cpl))):
                for k, v in errors(s, lines, lang, cpl).items():
                    tot[name][k] = tot[name].get(k, 0) + v
        results[lang] = {"sentences": len(sents), "cpl": cpl, **tot}
        print(lang, results[lang])
    save_json(data_dir() / "results" / "subs_layout.json", results)


if __name__ == "__main__":
    main()
