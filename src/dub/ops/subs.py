"""Subtitles: lay lines out per writing system, split long lines into cues, time the cues.

Layout (see `dub.ops.script`): lines break only at the break opportunities of the script
(spaces; BudouX phrases for Chinese/Japanese; dictionary words for Thai), widths are counted
as the style guides count them, and cues hold at most `max_lines` lines of `cpl` characters.

Timing: every item is one dubbed (or original) line with its time on the timeline. When the
item carries timed words of its audio, cue boundaries inside the line follow the voice:
each boundary is placed in the pause between the last word of one cue and the first word of
the next. Without words, time is shared by length. Afterwards cues may linger up to
`linger_ms` after the voice stops (easier to read), keep a `gap_ms` gap before the next cue,
and are stretched to `min_cue_ms` when there is room.
"""

from __future__ import annotations

import functools
import unicodedata

from dub.contracts.v1 import Cue, SubItem, SubsReq, SubtitleTrack
from dub.ops.align import char_times
from dub.ops.script import (
    CLAUSE_END,
    SENTENCE_END,
    break_tokens,
    display_len,
    ends_with,
    join_line,
    split_wide,
)
from dub.store.cas import CAS

_PUNCT = SENTENCE_END | CLAUSE_END


def _width(tokens: list[str], lang: str) -> float:
    return display_len("".join(tokens).strip(), lang)


def balanced_lines(tokens: list[str], cpl: int, lang: str, max_lines: int) -> list[str] | None:
    """Lay tokens out on <= max_lines lines of <= cpl width, with lines of similar width.

    Returns None if the tokens do not fit.
    """
    out = _balanced(tuple(tokens), cpl, lang, max_lines)
    return list(out) if out is not None else None


@functools.lru_cache(maxsize=1 << 15)
def _balanced(tokens: tuple[str, ...], cpl: int, lang: str, max_lines: int) -> tuple | None:
    """balanced_lines, memoised: the cue search asks for the same token runs over and over
    (a 60-minute speech took 12 s to lay out before)."""
    tokens = list(tokens)
    if _width(tokens, lang) <= cpl:
        return (join_line(tokens, lang),)
    if max_lines < 2 or len(tokens) < 2:
        return None
    best: tuple[float, list[str]] | None = None
    for i in range(1, len(tokens)):
        first = tokens[:i]
        w_first = _width(first, lang)
        if w_first > cpl:
            break
        rest = _balanced(tuple(tokens[i:]), cpl, lang, max_lines - 1)
        if rest is None:
            continue
        widths = [w_first] + [display_len(x, lang) for x in rest]
        score = max(widths) - (3 if ends_with(tokens[i - 1], _PUNCT) else 0)
        if best is None or score < best[0]:
            best = (score, (join_line(first, lang), *rest))
    return best[1] if best else None


def _fits(tokens: list[str], cpl: int, lang: str, max_lines: int) -> bool:
    """Can the tokens be laid out on <= max_lines lines of <= cpl? Filling each line as far as
    it goes is optimal for this yes/no question (balanced_lines then picks the nicest
    layout), and it is linear instead of a search."""
    if _width(tokens, lang) > cpl * max_lines:
        return False
    lines, start = 0, 0
    while start < len(tokens):
        lines += 1
        if lines > max_lines:
            return False
        end = start + 1
        if _width(tokens[start:end], lang) > cpl:
            return False
        while end < len(tokens) and _width(tokens[start : end + 1], lang) <= cpl:
            end += 1
        start = end
    return True


NO_PUNCT_PENALTY = 0.6  # cost of a cue break that is not after punctuation


def split_into_cues(tokens: list[str], cpl: int, lang: str, max_lines: int) -> list[list[str]]:
    """Split one long line into the fewest cues that fit, using dynamic programming:
    every cue must fit in max_lines x cpl; among those splits, prefer cues of similar size
    and breaks right after punctuation."""
    n = len(tokens)
    total = _width(tokens, lang)
    fits: dict[tuple[int, int], bool] = {}

    def ok(i: int, j: int) -> bool:
        if (i, j) not in fits:
            fits[(i, j)] = _fits(tokens[i:j], cpl, lang, max_lines)
        return fits[(i, j)]

    k_min = max(1, int(-(-total // (cpl * max_lines))))  # ceil
    inf = float("inf")
    for k in range(k_min, n + 1):
        target = total / k
        # best[c][j]: min cost covering tokens[:j] with c cues
        best = [[inf] * (n + 1) for _ in range(k + 1)]
        back = [[0] * (n + 1) for _ in range(k + 1)]
        best[0][0] = 0.0
        for c in range(1, k + 1):
            for j in range(1, n + 1):
                # shortest cue first: once tokens[i:j] does not fit, no longer cue does
                for i in range(j - 1, c - 2, -1):
                    if not ok(i, j):
                        break
                    if best[c - 1][i] == inf:
                        continue
                    size = _width(tokens[i:j], lang)
                    cost = ((size - target) / target) ** 2
                    if j < n and not ends_with(tokens[j - 1], _PUNCT):
                        cost += NO_PUNCT_PENALTY
                    if best[c - 1][i] + cost < best[c][j]:
                        best[c][j], back[c][j] = best[c - 1][i] + cost, i
        if best[k][n] < inf:
            chunks, j = [], n
            for c in range(k, 0, -1):
                i = back[c][j]
                chunks.append(tokens[i:j])
                j = i
            return chunks[::-1]
    return [[t] for t in tokens]


def _chunk_times(
    item: SubItem, text: str, chunks: list[list[str]], lang: str
) -> list[tuple[int, int]]:
    """(start, end) of the voice for each chunk of one item."""
    span = max(item.end_ms - item.start_ms, 1)
    times = char_times(text, item.words) if item.words else None
    if times is None:
        sizes = [max(_width(c, lang), 1.0) for c in chunks]
        total = sum(sizes)
        out, t = [], float(item.start_ms)
        for s in sizes:
            out.append((round(t), round(t + span * s / total)))
            t += span * s / total
        return out
    out, pos = [], 0
    for c in chunks:
        length = len("".join(c))
        a, b = pos, max(pos, pos + length - 1)
        # Skip leading/trailing spaces and punctuation inside the chunk for the voice times.
        while a < b and not text[a].isalnum():
            a += 1
        while b > a and not text[b].isalnum():
            b -= 1
        out.append((times[a][0], times[b][1]))  # type: ignore[index]
        pos += length
    # Boundaries between chunks go in the middle of the pause between them; the first and
    # last edges are the item's own edges (the clip was trimmed to the voice).
    bounds = [item.start_ms]
    for (_, e), (s, _) in zip(out, out[1:], strict=False):
        mid = (e + s) // 2 if s >= e else s
        bounds.append(min(max(mid, bounds[-1] + 1), item.end_ms - 1))
    bounds.append(item.end_ms)
    return [(bounds[i], bounds[i + 1]) for i in range(len(chunks))]


def build_cues(req: SubsReq) -> list[Cue]:
    out: list[Cue] = []
    for item in sorted(req.items, key=lambda c: c.start_ms):
        text = unicodedata.normalize("NFC", " ".join(item.text.split()))
        tokens = split_wide(break_tokens(text, req.lang), req.cpl, req.lang)
        if not tokens:
            continue
        chunks = split_into_cues(tokens, req.cpl, req.lang, req.max_lines)
        for chunk, (a, b) in zip(chunks, _chunk_times(item, text, chunks, req.lang), strict=True):
            lines = balanced_lines(chunk, req.cpl, req.lang, req.max_lines) or [
                join_line(chunk, req.lang)
            ]
            out.append(Cue(start_ms=a, end_ms=b, text="\n".join(lines)))
    out.sort(key=lambda c: c.start_ms)
    for k, c in enumerate(out):
        nxt = out[k + 1].start_ms - req.gap_ms if k + 1 < len(out) else None
        want = max(c.end_ms + req.linger_ms, c.start_ms + req.min_cue_ms)
        c.end_ms = max(c.end_ms, min(want, nxt) if nxt is not None else want)
        if nxt is not None and c.end_ms > out[k + 1].start_ms:
            c.end_ms = out[k + 1].start_ms  # cues never overlap
    return [c for c in out if c.end_ms > c.start_ms]


def _ts(ms: int) -> str:
    h, rem = divmod(max(ms, 0), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    return "\n".join(
        f"{i}\n{_ts(c.start_ms)} --> {_ts(c.end_ms)}\n{c.text}\n" for i, c in enumerate(cues, 1)
    )


def cps_violations(cues: list[Cue], cps: float, lang: str = "und") -> int:
    """Number of cues read much faster than the comfortable characters-per-second rate."""
    bad = 0
    for c in cues:
        chars = display_len(c.text.replace("\n", ""), lang)
        if chars / max((c.end_ms - c.start_ms) / 1000, 0.001) > cps * 1.5:
            bad += 1
    return bad


def render(req: SubsReq, cas: CAS) -> SubtitleTrack:
    cues = build_cues(req)
    key = cas.put_text(to_srt(cues), ".srt")
    return SubtitleTrack(lang=req.lang, cues=cues, srt_key=key)
