"""Script-aware text helpers for subtitles: display width, line-break opportunities, direction.

Writing systems disagree on what a "character" and a "word" are:
- Latin, Cyrillic, Vietnamese, Korean... separate words with spaces: break at spaces.
- Chinese and Japanese have no spaces: we break between BudouX phrases (Google's small
  phrase model) and follow the basic kinsoku rules (no line starts with "。" or "」", none
  ends with "「").
- Thai has no spaces between words either. A break is allowed only where BOTH PyThaiNLP
  (dictionary word segmentation) and BudouX see a boundary, so neither tool alone can split a
  compound word such as "วันนี้" ("today"). A zero-width space (U+200B) is inserted at those
  points so players that re-wrap lines also break correctly.
- Arabic and Hebrew run right to left: each line starts with a RIGHT-TO-LEFT MARK so that
  players which guess the direction from the first strong character get it right.

Widths follow the Netflix style guides: combining marks (Thai tone marks, vowels above and
below) take no space, and in Chinese/Japanese/Korean a half-width character counts as half.
"""

from __future__ import annotations

import functools
import re
import unicodedata

NO_SPACE_LANGS = {"zh", "ja", "th", "lo", "km", "my", "yue", "bo"}
CJK_LANGS = {"zh", "ja", "ko", "yue"}
ZWSP_LANGS = {"th", "lo", "km", "my"}  # renderers cannot find word breaks on their own

ZWSP = "​"
RLM = "‏"

# Kinsoku shori: characters that may not start a line, and that may not end one.
NO_START = set(
    "、。，．・：；？！ー―〜…‥）〕］｝〉》」』】〙〗〟’”»"
    "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ々〻"
    ",.;:!?)]}%"
)
NO_END = set("（〔［｛〈《「『【〘〖〝‘“«([{")
SENTENCE_END = set(".?!。？！…؟।")
CLAUSE_END = set(",;:，、；：")


def base_lang(lang: str) -> str:
    return (lang or "").split("-")[0].split("_")[0].lower()


def is_no_space(lang: str) -> bool:
    return base_lang(lang) in NO_SPACE_LANGS


def display_len(s: str, lang: str) -> float:
    """Width of a subtitle line in "characters", as the style guides count them."""
    cjk = base_lang(lang) in CJK_LANGS
    width = 0.0
    for ch in s:
        if unicodedata.category(ch) in ("Mn", "Me", "Cf"):
            continue  # combining marks and invisible format characters (ZWSP, RLM...)
        if cjk:
            width += 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5
        else:
            width += 1.0
    return width


def is_rtl(text: str) -> bool:
    """True if most strongly-directional letters are right-to-left (Arabic, Hebrew...)."""
    rtl = ltr = 0
    for ch in text:
        b = unicodedata.bidirectional(ch)
        if b in ("R", "AL"):
            rtl += 1
        elif b == "L":
            ltr += 1
    return rtl > ltr


# ---------------------------------------------------------------- break opportunities


@functools.cache
def _budoux(lang: str):
    import budoux

    if lang == "ja":
        return budoux.load_default_japanese_parser()
    if lang == "th":
        return budoux.load_default_thai_parser()
    if lang in ("zh-tw", "zh-hk", "zh-hant", "yue"):
        return budoux.load_default_traditional_chinese_parser()
    return budoux.load_default_simplified_chinese_parser()


def _cuts(pieces: list[str]) -> set[int]:
    """Character offsets between consecutive pieces."""
    out, pos = set(), 0
    for p in pieces[:-1]:
        pos += len(p)
        out.add(pos)
    return out


def _thai_cuts(chunk: str) -> set[int]:
    budoux_cuts = _cuts(_budoux("th").parse(chunk))
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:  # pragma: no cover - pythainlp is a project dependency
        return budoux_cuts
    return budoux_cuts & _cuts(word_tokenize(chunk, engine="newmm", keep_whitespace=True))


def _split_chunk(chunk: str, lang: str) -> list[str]:
    """Split one whitespace-free chunk into unbreakable pieces."""
    b = base_lang(lang)
    if b == "th":
        cuts = sorted(_thai_cuts(chunk))
    elif b in ("zh", "ja", "yue"):
        key = lang.lower() if b == "zh" else b
        cuts = sorted(_cuts(_budoux(key).parse(chunk)))
    elif b in NO_SPACE_LANGS:
        cuts = list(range(1, len(chunk)))  # no dictionary: any character boundary
    else:
        return [chunk]
    out, prev = [], 0
    for c in cuts:
        out.append(chunk[prev:c])
        prev = c
    out.append(chunk[prev:])
    return [p for p in out if p]


def _kinsoku(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if out and (tok[0] in NO_START or out[-1].rstrip()[-1:] in NO_END):
            out[-1] += tok
        else:
            out.append(tok)
    return out


def break_tokens(text: str, lang: str) -> list[str]:
    """Split text into the smallest pieces a line may break between.

    Each piece keeps its trailing whitespace, so "".join(pieces) == text (after collapsing
    runs of whitespace). Space-separated scripts give one piece per word.
    """
    text = " ".join(text.split())
    if not text:
        return []
    tokens: list[str] = []
    for m in re.finditer(r"(\S+)(\s*)", text):
        pieces = _split_chunk(m.group(1), lang)
        pieces[-1] += m.group(2)
        tokens.extend(pieces)
    return _kinsoku(tokens) if is_no_space(lang) else tokens


def split_wide(tokens: list[str], cpl: int, lang: str) -> list[str]:
    """Split pieces wider than a whole line (a long katakana word, a URL) by character, still
    following kinsoku. Breaking inside them is the lesser evil versus an over-wide line."""
    out: list[str] = []
    for tok in tokens:
        if display_len(tok.strip(), lang) <= cpl:
            out.append(tok)
            continue
        out.extend(_kinsoku(list(tok.rstrip())))
        if tok != tok.rstrip():
            out[-1] += tok[len(tok.rstrip()) :]
    return out


def join_line(tokens: list[str], lang: str) -> str:
    """Join tokens into one display line (adds ZWSP for Thai-like scripts, RLM for RTL)."""
    if base_lang(lang) in ZWSP_LANGS:
        parts = [t if t.endswith(" ") else t + ZWSP for t in tokens[:-1]] + tokens[-1:]
        line = "".join(parts).strip()
    else:
        line = "".join(tokens).strip()
    return RLM + line if is_rtl(line) else line


def ends_with(token: str, chars: set[str]) -> bool:
    s = token.rstrip().rstrip("\"'”’»)]}」』）")
    return s[-1:] in chars
