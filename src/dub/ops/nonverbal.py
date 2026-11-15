"""Non-verbal vocal sounds kept in the dub: laughter, crying, screams, gasps, sighs (M10).

The separator puts every sound a person makes into the vocals stem, and the dub replaces that
stem with the dubbed lines. So a laugh, a scream or a sob that the ASR did not transcribe as a
line was simply lost (the cartoon: a 3.3 s laugh and the last 1.8 s of a scream; the drama: a
baby's cry past its 0.7 s line).

Keeping every vocal sound between the lines would leak the source language too (an answer the
ASR missed: 你瘦了 between two lines of the drama). So a vocal region between lines is kept
only when it carries no words:
1. `gaps`: where the vocals stem has sound (energy VAD) but no line and no subtitle cue is
   (each line padded a little: its slot is where the dub speaks);
2. the source-language ASR listens to each gap: no words, or only laughter and cries (the
   langpack's `interjections`: 哈哈, 啊...), keeps it; words drop it;
3. the kept regions are mixed as they are (original audio), like a line kept original.
"""

from __future__ import annotations


def subtract(spans: list[tuple[int, int]], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The parts of `spans` outside every interval of `cuts`."""
    out: list[tuple[int, int]] = []
    cuts = sorted(cuts)
    for s, e in spans:
        cur = s
        for a, b in cuts:
            if b <= cur or a >= e:
                continue
            if a > cur:
                out.append((cur, a))
            cur = max(cur, b)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e))
    return out


def gaps(
    regions: list[tuple[int, int]],
    busy: list[tuple[int, int]],
    *,
    pad_ms: int = 150,
    merge_ms: int = 200,
    min_ms: int = 300,
) -> list[tuple[int, int]]:
    """Vocal sound outside the busy spans (lines, subtitle cues), each busy span padded by
    `pad_ms`; pieces closer than `merge_ms` are joined, shorter than `min_ms` dropped."""
    padded = [(max(0, a - pad_ms), b + pad_ms) for a, b in busy]
    free = sorted(subtract(sorted(regions), padded))
    merged: list[tuple[int, int]] = []
    for s, e in free:
        if merged and s - merged[-1][1] <= merge_ms:
            # do not join across a busy span: only pieces whose gap is itself free
            between = (merged[-1][1], s)
            if not subtract([between], padded) == [between]:
                merged.append((s, e))
                continue
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_ms]


def wordless(text: str, is_interjection) -> bool:
    """What the ASR heard in a gap holds no words: nothing, or only laughter and cries."""
    t = (text or "").strip()
    return not t or bool(is_interjection(t))


def _content(text: str) -> list[str]:
    return [ch for ch in text if ch.isalnum()]


def share(cue_text: str, line_text: str) -> float:
    """How much of a subtitle cue's text is in a line (characters, in any order)."""
    cue = _content(cue_text)
    if not cue:
        return 0.0
    pool = _content(line_text)
    hit = 0
    for ch in cue:
        if ch in pool:
            pool.remove(ch)
            hit += 1
    return hit / len(cue)


def original_extents(
    lines: list[tuple[str, int, int, str]],
    original: set[str],
    cues: list[tuple[int, int, str]],
    regions: list[tuple[int, int]],
    *,
    pad_ms: int = 150,
    vad_margin_ms: int = 800,
    max_extend_ms: int = 6000,
    min_share: float = 0.6,
    min_chars: int = 2,
) -> dict[str, tuple[int, int]]:
    """Where a line kept in the original audio really sounds: its slot, the subtitle cues
    that carry its text, and the vocal sound touching them, but never into a dubbed line's
    slot (padded) nor into another kept line. The ASR had squeezed a 5 s song into 300 ms
    (word times collapsed): cut at the slot, the song went silent from 0.3 s to 3.8 s. Only
    sound overlapping the line or its cues is taken: anything beside it goes through the
    non-verbal step and its check for words.

    `lines`: (id, start_ms, end_ms, source text) in time order; `cues`: (start, end, text)."""
    lines = sorted(lines, key=lambda x: (x[1], x[2]))
    out: dict[str, tuple[int, int]] = {}
    for i, (uid, s0, e0, text) in enumerate(lines):
        if uid not in original:
            continue
        lo = max((e + pad_ms for v, _, e, _ in lines[:i] if v not in original), default=0)
        hi = min((s - pad_ms for v, s, _, _ in lines[i + 1:] if v not in original),
                 default=e0 + max_extend_ms)  # fmt: skip
        a, b = s0, e0
        for c0, c1, ct in cues:
            near = c1 >= s0 - max_extend_ms and c0 <= e0 + max_extend_ms
            # most of the cue's characters, and at least two, are the line's ("大爷" shares
            # one character with 扭扭大肚肚: the next line's cue, not the song's)
            sh = share(ct, text)
            if near and sh >= min_share and sh * len(_content(ct)) >= min_chars:
                a, b = min(a, c0), max(b, c1)
        core = (a, b)
        # the vocal sound overlapping the line and its cues, at most vad_margin_ms beyond them:
        # a separate sound next to it (a word of the next line) is the non-verbal step's to judge,
        # with its leak check (first version took in a word 250 ms after the song)
        for r0, r1 in regions:
            if r0 < core[1] and r1 > core[0]:
                a, b = min(a, r0), max(b, r1)
        a = max(a, core[0] - vad_margin_ms, lo, s0 - max_extend_ms)
        b = min(b, core[1] + vad_margin_ms, hi, e0 + max_extend_ms)
        out[uid] = (min(a, s0), max(b, e0)) if b > a else (s0, e0)
    # two kept lines never sound twice over the same span: each starts where the last ended
    done = 0
    for uid in sorted(out, key=lambda k: out[k]):
        a, b = out[uid]
        a = max(a, done)
        out[uid] = (a, max(a, b))
        done = max(done, out[uid][1])
    return out
