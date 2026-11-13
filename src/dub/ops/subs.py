"""Subtitles: wrap lines by characters per line (CPL), split long cues, write SRT.

M0 wraps on spaces (space-delimited languages) or on characters (zh/ja). Thai needs
dictionary-based word breaking (pythainlp/ICU), planned for M3.
"""

from __future__ import annotations

from dub.contracts.v1 import Cue, SubsReq, SubtitleTrack
from dub.store.cas import CAS


def _tokens(text: str, joiner: str) -> list[str]:
    return text.split() if joiner == " " else list(text.replace(" ", ""))


def wrap_lines(text: str, cpl: int, joiner: str) -> list[str]:
    lines: list[str] = []
    cur: list[str] = []
    for tok in _tokens(text, joiner):
        cand = joiner.join(cur + [tok])
        if cur and len(cand) > cpl:
            lines.append(joiner.join(cur))
            cur = [tok]
        else:
            cur.append(tok)
    if cur:
        lines.append(joiner.join(cur))
    return lines


_BREAK_AFTER = set(",.;:!?，。；：！？、")


def balanced_lines(tokens: list[str], cpl: int, joiner: str, max_lines: int) -> list[str] | None:
    """Lay tokens out on <= max_lines lines of <= cpl chars, with lines of similar length.

    Returns None if the tokens do not fit.
    """
    one = joiner.join(tokens)
    if len(one) <= cpl:
        return [one]
    if max_lines < 2 or len(tokens) < 2:
        return None
    best: tuple[float, list[str]] | None = None
    for i in range(1, len(tokens)):
        first = joiner.join(tokens[:i])
        if len(first) > cpl:
            break
        rest = balanced_lines(tokens[i:], cpl, joiner, max_lines - 1)
        if rest is None:
            continue
        lines = [first, *rest]
        score = max(len(x) for x in lines) - (3 if tokens[i - 1][-1:] in _BREAK_AFTER else 0)
        if best is None or score < best[0]:
            best = (score, lines)
    return best[1] if best else None


NO_PUNCT_PENALTY = 0.6  # cost of a cue break that is not after punctuation


def split_into_cues(tokens: list[str], cpl: int, joiner: str, max_lines: int) -> list[list[str]]:
    """Split one long line into the fewest cues that fit, using dynamic programming:
    every cue must fit in max_lines x cpl; among those splits, prefer cues of similar size
    and breaks right after punctuation."""
    n = len(tokens)
    total = len(joiner.join(tokens))
    fits = {}

    def ok(i: int, j: int) -> bool:
        if (i, j) not in fits:
            fits[(i, j)] = balanced_lines(tokens[i:j], cpl, joiner, max_lines) is not None
        return fits[(i, j)]

    k_min = max(1, -(-total // (cpl * max_lines)))  # ceil
    for k in range(k_min, n + 1):
        target = total / k
        inf = float("inf")
        # best[c][j]: min cost covering tokens[:j] with c cues
        best = [[inf] * (n + 1) for _ in range(k + 1)]
        back = [[0] * (n + 1) for _ in range(k + 1)]
        best[0][0] = 0.0
        for c in range(1, k + 1):
            for j in range(1, n + 1):
                for i in range(c - 1, j):
                    if best[c - 1][i] == inf or not ok(i, j):
                        continue
                    size = len(joiner.join(tokens[i:j]))
                    cost = ((size - target) / target) ** 2
                    if j < n and tokens[j - 1][-1:] not in _BREAK_AFTER:
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


def build_cues(req: SubsReq) -> list[Cue]:
    out: list[Cue] = []
    for item in sorted(req.items, key=lambda c: c.start_ms):
        tokens = _tokens(" ".join(item.text.split()), req.joiner)
        if not tokens:
            continue
        chunks = split_into_cues(tokens, req.cpl, req.joiner, req.max_lines)
        sizes = [len(req.joiner.join(c)) for c in chunks]
        total_chars = sum(sizes) or 1
        t = item.start_ms
        span = max(item.end_ms - item.start_ms, 1)
        for chunk, size in zip(chunks, sizes, strict=True):
            end = t + round(span * size / total_chars)
            lines = balanced_lines(chunk, req.cpl, req.joiner, req.max_lines) or [
                req.joiner.join(chunk)
            ]
            out.append(Cue(start_ms=t, end_ms=end, text="\n".join(lines)))
            t = end
    for a, b in zip(out, out[1:], strict=False):  # cues must not overlap
        if a.end_ms > b.start_ms:
            a.end_ms = b.start_ms
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


def cps_violations(cues: list[Cue], cps: float) -> int:
    """Number of cues read much faster than the comfortable characters-per-second rate."""
    bad = 0
    for c in cues:
        chars = len(c.text.replace("\n", ""))
        if chars / max((c.end_ms - c.start_ms) / 1000, 0.001) > cps * 1.5:
            bad += 1
    return bad


def render(req: SubsReq, cas: CAS) -> SubtitleTrack:
    cues = build_cues(req)
    key = cas.put_text(to_srt(cues), ".srt")
    return SubtitleTrack(lang=req.lang, cues=cues, srt_key=key)
