"""Segmentation: timed words -> sentences -> dub units.

This is the project's #1 priority. The M0 version is a rule + dynamic-programming baseline;
M3 replaces the boundary scores with SaT model probabilities but keeps the algorithm.

Three layers of boundaries:
1. Mandatory: speaker change, or a pause >= hard_pause (1.5 s). No unit crosses one.
2. Sentence: after terminal punctuation (. ? ! 。 ？ ！ …).
3. Over-long sentences (> max_unit, 15 s) are split with dynamic programming, preferring
   cuts at commas and long pauses, and keeping each unit close to target_unit (7 s).
Finally, units shorter than min_unit (1 s) are merged into a neighbour of the same speaker.
"""

from __future__ import annotations

from dub.contracts.v1 import (
    Segmentation,
    SegmentParams,
    SegmentReq,
    Sentence,
    Turn,
    Unit,
    Word,
)

TERMINAL = set(".?!。？！…؟।")
SOFT = set(",;:，、；：—–")
CLOSERS = "\"'”’»)]}」』"
INF = float("inf")
# A period after these does not end a sentence ("Mr. Tu", "Dr. Lee", "e.g. this").
ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "mt",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "no",
    "fig",
    "approx",
    "sra",
    "srta",
    "m",
    "mme",
    "mlle",
    "hr",
    "fr",
}


def _last_char(w: str) -> str:
    s = w.rstrip(CLOSERS)
    return s[-1:] if s else ""


def _ends_sentence(w: str) -> bool:
    last = _last_char(w)
    if last not in TERMINAL:
        return False
    if last == ".":
        stem = w.rstrip(CLOSERS)[:-1].lower()
        if stem in ABBREVIATIONS or (len(stem) == 1 and stem.isalpha()):
            return False  # abbreviation or an initial ("J. Smith")
    return True


def assign_speakers(words: list[Word], turns: list[Turn]) -> list[Word]:
    """Give each word the speaker of the turn it overlaps most (or the nearest turn)."""
    if not turns:
        return [w.model_copy(update={"speaker": w.speaker or "S1"}) for w in words]
    out = []
    for w in words:
        best, best_ov, best_dist = None, 0, INF
        for t in turns:
            ov = min(w.end_ms, t.end_ms) - max(w.start_ms, t.start_ms)
            if ov > best_ov:
                best, best_ov = t.speaker, ov
            elif best_ov == 0:
                dist = max(t.start_ms - w.end_ms, w.start_ms - t.end_ms)
                if dist < best_dist:
                    best, best_dist = t.speaker, dist
        out.append(w.model_copy(update={"speaker": best}))
    return out


def _blocks(words: list[Word], p: SegmentParams) -> list[list[Word]]:
    blocks: list[list[Word]] = []
    for w in words:
        if blocks:
            prev = blocks[-1][-1]
            if w.speaker == prev.speaker and w.start_ms - prev.end_ms < p.hard_pause_ms:
                blocks[-1].append(w)
                continue
        blocks.append([w])
    return blocks


def _sentences(block: list[Word]) -> list[list[Word]]:
    out: list[list[Word]] = [[]]
    for w in block:
        out[-1].append(w)
        if _ends_sentence(w.w):
            out.append([])
    return [s for s in out if s]


def _unit_cost(dur: int, p: SegmentParams) -> float:
    x = (dur - p.target_unit_ms) / p.target_unit_ms
    cost = 2.0 * x * x
    if dur < p.min_unit_ms:
        cost += 4.0
    return cost


def _cut_cost(ws: list[Word], i: int) -> float:
    """Cost of cutting between ws[i-1] and ws[i]. Lower means a better place to cut."""
    base = 0.5 if _last_char(ws[i - 1].w) in SOFT else 3.0
    gap = ws[i].start_ms - ws[i - 1].end_ms
    bonus = min(2.0, gap / 250.0)  # a 0.5 s pause takes 2 points off
    return max(0.1, base - bonus) + 1.0  # +1 per cut so we do not over-split


def split_long(ws: list[Word], p: SegmentParams) -> list[list[Word]]:
    """Split one long sentence into pieces <= max_unit_ms using dynamic programming.

    best[j] = lowest cost to cover ws[:j]; the last piece is ws[i:j].
    """
    if len(ws) <= 1 or ws[-1].end_ms - ws[0].start_ms <= p.max_unit_ms:
        return [ws]
    n = len(ws)
    best = [0.0] + [INF] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            dur = ws[j - 1].end_ms - ws[i].start_ms
            if dur > p.max_unit_ms and j - i > 1:
                break  # moving i further left only makes the piece longer
            if best[i] == INF:
                continue
            c = best[i] + _unit_cost(dur, p) + (_cut_cost(ws, i) if i > 0 else 0.0)
            if c < best[j]:
                best[j], back[j] = c, i
    chunks, j = [], n
    while j > 0:
        i = back[j]
        chunks.append(ws[i:j])
        j = i
    return chunks[::-1]


def _merge_tiny(chunks: list[tuple[list[Word], list[str]]], p: SegmentParams):
    """Merge units shorter than min_unit_ms into a neighbour (same block => same speaker)."""

    def dur(ws: list[Word]) -> int:
        return ws[-1].end_ms - ws[0].start_ms

    changed = True
    while changed and len(chunks) > 1:
        changed = False
        for k, (ws, _sids) in enumerate(chunks):
            if dur(ws) >= p.min_unit_ms:
                continue
            for nb in (k + 1, k - 1):  # prefer merging with the following unit
                if 0 <= nb < len(chunks):
                    a, b = sorted((k, nb))
                    merged = chunks[a][0] + chunks[b][0]
                    if dur(merged) <= p.max_unit_ms:
                        sids_m = chunks[a][1] + [s for s in chunks[b][1] if s not in chunks[a][1]]
                        chunks[a : b + 1] = [(merged, sids_m)]
                        changed = True
                        break
            if changed:
                break
    return chunks


def _join(ws: list[Word], joiner: str) -> str:
    return joiner.join(w.w.strip() for w in ws).strip()


def segment(req: SegmentReq) -> Segmentation:
    p = req.params
    words = sorted((w for w in req.words if w.w.strip()), key=lambda w: (w.start_ms, w.end_ms))
    words = assign_speakers(words, req.turns)
    sentences: list[Sentence] = []
    units: list[Unit] = []
    for block in _blocks(words, p):
        chunks: list[tuple[list[Word], list[str]]] = []
        for sw in _sentences(block):
            sid = f"s{sw[0].start_ms:07d}-{sw[0].speaker}"
            sentences.append(
                Sentence(
                    id=sid,
                    speaker=sw[0].speaker or "S1",
                    start_ms=sw[0].start_ms,
                    end_ms=sw[-1].end_ms,
                    text=_join(sw, p.joiner),
                )
            )
            chunks.extend((c, [sid]) for c in split_long(sw, p))
        for ws, sids in _merge_tiny(chunks, p):
            spk = ws[0].speaker or "S1"
            units.append(
                Unit(
                    id=f"u{ws[0].start_ms:07d}-{spk}",
                    speaker=spk,
                    sentence_ids=sids,
                    start_ms=ws[0].start_ms,
                    end_ms=ws[-1].end_ms,
                    text=_join(ws, p.joiner),
                    n_words=len(ws),
                    asr_conf=round(sum(w.conf for w in ws) / len(ws), 3),
                )
            )
    units.sort(key=lambda u: (u.start_ms, u.speaker))
    return Segmentation(lang=req.lang, sentences=sentences, units=units)
