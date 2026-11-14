"""Text <-> timed words: character alignment and character error rate (CER).

Two uses:
- Subtitles that follow the voice. The dubbed clip is transcribed again (the "back-check"),
  and every character of the line we asked the TTS to say is mapped to a time through the
  recognised words. Cue boundaries inside a line then land where the voice really is,
  instead of being spread by character count.
- Quality: the CER between the line and what the back-check heard is a cheap signal that
  the TTS skipped, slurred or invented words.

Matching ignores case, punctuation and (for alignment only) diacritics, so a tone the ASR
got wrong still anchors the time.
"""

from __future__ import annotations

import bisect
import difflib
import unicodedata

from dub.contracts.v1 import Word

Times = tuple[int, int]


def _fold(ch: str, strip_marks: bool) -> str:
    if strip_marks and ch in "đĐ":
        ch = "d"
    ch = ch.casefold()
    if strip_marks:
        ch = "".join(c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", ch)


def _norm_chars(text: str, strip_marks: bool) -> list[tuple[int, str]]:
    """(index in text, folded char) for every letter or digit of text."""
    out = []
    for i, ch in enumerate(unicodedata.normalize("NFC", text)):
        if ch.isalnum():
            out.extend((i, f) for f in _fold(ch, strip_marks))
    return out


def char_times(text: str, words: list[Word]) -> list[Times | None] | None:
    """For each character of `text`, the (start_ms, end_ms) at which it is spoken.

    `words` are timed ASR words of the audio that says `text`. Characters the ASR did not
    match are interpolated between their matched neighbours. Returns None when nothing
    matches at all (then callers fall back to spreading time by length).
    """
    text = unicodedata.normalize("NFC", text)
    ref = _norm_chars(text, strip_marks=True)
    hyp: list[tuple[str, Times]] = []
    for w in words:
        chars = [c for _, c in _norm_chars(w.w, strip_marks=True)]
        span = max(w.end_ms - w.start_ms, 0)
        for k, c in enumerate(chars):
            s = w.start_ms + span * k / len(chars)
            e = w.start_ms + span * (k + 1) / len(chars)
            hyp.append((c, (round(s), round(e))))
    if not ref or not hyp:
        return None
    sm = difflib.SequenceMatcher(None, [c for _, c in ref], [c for c, _ in hyp], autojunk=False)
    anchor: dict[int, Times] = {}  # index in ref -> time
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            anchor[a + k] = hyp[b + k][1]
    if not anchor:
        return None

    # Interpolate the unmatched letters between matched ones.
    keys = sorted(anchor)
    ref_t: list[Times] = []
    for r in range(len(ref)):
        if r in anchor:
            ref_t.append(anchor[r])
            continue
        pos = bisect.bisect_left(keys, r)
        before = keys[pos - 1] if pos > 0 else None
        after = keys[pos] if pos < len(keys) else None
        if before is None:
            ref_t.append((anchor[after][0], anchor[after][0]))
        elif after is None:
            ref_t.append((anchor[before][1], anchor[before][1]))
        else:
            t0, t1 = anchor[before][1], anchor[after][0]
            f = (r - before) / (after - before)
            t = round(t0 + (t1 - t0) * f)
            ref_t.append((t, t))

    # Spread to every character of the text (punctuation and spaces take the time of the
    # letter before them).
    out: list[Times | None] = [None] * len(text)
    for (i, _), t in zip(ref, ref_t, strict=True):
        out[i] = t
    last: Times | None = None
    for i in range(len(text)):
        if out[i] is None:
            out[i] = (last[1], last[1]) if last else None
        else:
            last = out[i]
    first = next((t for t in out if t is not None), None)
    return [t if t is not None else (first[0], first[0]) for t in out]  # type: ignore[index]


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate, ignoring case, punctuation and spaces (diacritics count)."""
    r = [c for _, c in _norm_chars(ref, strip_marks=False)]
    h = [c for _, c in _norm_chars(hyp, strip_marks=False)]
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)
