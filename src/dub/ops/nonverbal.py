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
