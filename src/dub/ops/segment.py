"""Segmentation: timed words -> sentences -> dub units.

This is the project's #1 priority. Every word carries a boundary probability: P(a sentence
ends right after this word). It comes from the `text.boundaries` op: the SaT model on the
pod, or `rule_probs` below (punctuation rules) as the fallback and in tests.

Three layers of boundaries:
1. Mandatory: speaker change, or a pause >= hard_pause (1.5 s). No unit crosses one. A
   sentence ends at a speaker change, but may continue across a long pause (an orator
   waiting for applause): the units are cut there, the sentence is not.
2. Sentence: where the boundary probability reaches `threshold`.
3. Over-long sentences (> max_unit, 15 s) are split with dynamic programming. A cut costs
   -log(p) minus a bonus for the pause there, so the DP prefers likely boundaries and long
   pauses, and keeps each unit close to target_unit (7 s).
Units shorter than min_unit (1 s) are merged into a neighbour of the same speaker.

Finally, unit edges are snapped to the speech regions found by a VAD on the separated
vocals: ASR word times are often 100-300 ms off, and the dub is placed on these edges.
"""

from __future__ import annotations

import bisect
import math

from dub.contracts.v1 import (
    Segmentation,
    SegmentParams,
    SegmentReq,
    Sentence,
    Turn,
    Unit,
    Word,
    join_words,
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


P_SENTENCE, P_CLAUSE, P_NONE = 0.95, 0.3, 0.02
_LEADING = ".?!。？！…"


def normalize_words(words: list[Word]) -> list[Word]:
    """Move punctuation an ASR glued to the START of a word back to the previous word.

    PhoWhisper sometimes writes "lo toan .đặc biệt": the sentence end belongs to "toan", but a
    boundary model reading ".đặc" would put it inside the next word, where nobody looks.
    """
    out: list[Word] = []
    for w in words:
        text = w.w
        lead = len(text) - len(text.lstrip(_LEADING))
        if lead and out and text[lead:].strip():
            prev = out[-1]
            out[-1] = prev.model_copy(update={"w": prev.w + text[:lead]})
            w = w.model_copy(update={"w": text[lead:]})
        out.append(w)
    return out


def rule_probs(words: list[str]) -> list[float]:
    """Boundary probabilities from punctuation alone (the pre-SaT baseline)."""
    out = []
    for w in words:
        if _ends_sentence(w):
            out.append(P_SENTENCE)
        elif _last_char(w) in SOFT:
            out.append(P_CLAUSE)
        else:
            out.append(P_NONE)
    return out


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


def smooth_speakers(words: list[Word], max_ms: int = 600, max_words: int = 2) -> list[Word]:
    """A word or two given to another speaker in the middle of one speaker's run is almost
    always a diarization slip at a turn edge: give them back (no unit is cut there)."""
    out = list(words)
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1].speaker == out[i].speaker:
            j += 1
        if 0 < i and j + 1 < len(out):
            before, after = out[i - 1].speaker, out[j + 1].speaker
            dur = out[j].end_ms - out[i].start_ms
            if before == after != out[i].speaker and j - i + 1 <= max_words and dur < max_ms:
                for k in range(i, j + 1):
                    out[k] = out[k].model_copy(update={"speaker": before})
        i = j + 1
    return out


def _blocks(words: list[Word], probs: list[float]):
    """Split at speaker changes: [(words, probs)] per block."""
    blocks: list[tuple[list[Word], list[float]]] = []
    for w, pr in zip(words, probs, strict=True):
        if blocks and w.speaker == blocks[-1][0][-1].speaker:
            blocks[-1][0].append(w)
            blocks[-1][1].append(pr)
        else:
            blocks.append(([w], [pr]))
    return blocks


def _pause_pieces(ws: list[Word], probs: list[float], p: SegmentParams):
    """Split a sentence at long pauses: [(words, probs, starts_after_pause)]."""
    pieces: list[tuple[list[Word], list[float], bool]] = []
    for i, (w, pr) in enumerate(zip(ws, probs, strict=True)):
        if i and w.start_ms - ws[i - 1].end_ms < p.hard_pause_ms:
            pieces[-1][0].append(w)
            pieces[-1][1].append(pr)
        else:
            pieces.append(([w], [pr], i > 0))
    return pieces


def _sentences(block: list[Word], probs: list[float], threshold: float):
    out: list[tuple[list[Word], list[float]]] = [([], [])]
    for w, pr in zip(block, probs, strict=True):
        out[-1][0].append(w)
        out[-1][1].append(pr)
        if pr >= threshold:
            out.append(([], []))
    return [s for s in out if s[0]]


def _unit_cost(dur: int, p: SegmentParams) -> float:
    x = (dur - p.target_unit_ms) / p.target_unit_ms
    cost = 2.0 * x * x
    if dur < p.min_unit_ms:
        cost += 4.0
    return cost


def _cut_cost(ws: list[Word], probs: list[float], i: int) -> float:
    """Cost of cutting between ws[i-1] and ws[i]. Lower means a better place to cut.

    -log(p): a clause boundary (p 0.3) costs 1.2, an unlikely place (p 0.02) 3.9.
    """
    base = min(4.0, -math.log(max(probs[i - 1], 0.02)))
    gap = ws[i].start_ms - ws[i - 1].end_ms
    bonus = min(2.0, gap / 250.0)  # a 0.5 s pause takes 2 points off
    return max(0.1, base - bonus) + 1.0  # +1 per cut so we do not over-split


def split_long(
    ws: list[Word], p: SegmentParams, probs: list[float] | None = None
) -> list[list[Word]]:
    """Split one long sentence into pieces <= max_unit_ms using dynamic programming.

    best[j] = lowest cost to cover ws[:j]; the last piece is ws[i:j].
    """
    if probs is None:
        probs = rule_probs([w.w for w in ws])
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
            c = best[i] + _unit_cost(dur, p) + (_cut_cost(ws, probs, i) if i > 0 else 0.0)
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


SNAP_IN_SPEECH_MS = 400  # an edge inside speech moves at most this far to the region's edge
SNAP_IN_SILENCE_MS = 800  # an edge in silence moves at most this far to the nearest speech
SNAP_PAD_MS = 30  # air kept before an onset and after an offset


def _region_at(speech: list[tuple[int, int]], t: int) -> tuple[int, int] | None:
    i = bisect.bisect_right(speech, (t, 1 << 62)) - 1
    if i >= 0 and speech[i][0] <= t < speech[i][1]:
        return speech[i]
    return None


def snap_start(t: int, speech: list[tuple[int, int]]) -> int:
    """Where the voice really starts, for an ASR start time t (or t if unsure)."""
    region = _region_at(speech, t)
    if region is not None:  # inside speech: the onset is at most a little earlier
        return region[0] - SNAP_PAD_MS if t - region[0] <= SNAP_IN_SPEECH_MS else t
    # In silence: no word starts there. Whisper puts first words early, into the pause.
    nxt = [a for a, _ in speech if t <= a <= t + SNAP_IN_SILENCE_MS]
    return nxt[0] - SNAP_PAD_MS if nxt else t


def snap_end(t: int, speech: list[tuple[int, int]]) -> int:
    """Where the voice really stops, for an ASR end time t (or t if unsure)."""
    region = _region_at(speech, t - 1)
    if region is not None:  # inside speech: the offset is at most a little later
        return region[1] + SNAP_PAD_MS if region[1] - t <= SNAP_IN_SPEECH_MS else t
    prev = [b for _, b in speech if t - SNAP_IN_SILENCE_MS <= b <= t]
    return prev[-1] + SNAP_PAD_MS if prev else t


def snap_to_speech(
    units: list[Unit], speech: list[tuple[int, int]], min_gap_ms: int = 60
) -> list[Unit]:
    """Move unit edges onto the speech found by a VAD on the separated vocals.

    ASR word times are often off by a few hundred ms; Whisper in particular starts the first
    word of a phrase 300-500 ms early, inside the pause before it (measured in M3). An edge
    that falls in silence moves to the nearest speech (up to 800 ms); an edge inside speech
    moves to the edge of that speech region if it is close (up to 400 ms). Edges never
    cross the neighbouring units, and a unit never shrinks below 200 ms.
    """
    if not speech:
        return units
    speech = sorted(speech)
    out: list[Unit] = []
    for k, u in enumerate(units):
        lo = out[-1].end_ms + min_gap_ms if out else 0
        hi = units[k + 1].start_ms - min_gap_ms if k + 1 < len(units) else None
        start = max(lo, snap_start(u.start_ms, speech))
        end = snap_end(u.end_ms, speech)
        if hi is not None:
            end = min(end, max(hi, u.end_ms))
        if end - start < 200:  # never collapse a unit: keep the ASR edges instead
            start, end = u.start_ms, u.end_ms
        out.append(u.model_copy(update={"start_ms": int(start), "end_ms": int(end)}))
    return out


def segment(req: SegmentReq) -> Segmentation:
    p = req.params
    order = sorted(
        (i for i, w in enumerate(req.words) if w.w.strip()),
        key=lambda i: (req.words[i].start_ms, req.words[i].end_ms),
    )
    words = normalize_words(
        smooth_speakers(assign_speakers([req.words[i] for i in order], req.turns))
    )
    if req.probs:
        if len(req.probs) != len(req.words):
            raise ValueError(f"{len(req.probs)} boundary probs for {len(req.words)} words")
        probs = [req.probs[i] for i in order]
    else:
        probs = rule_probs([w.w for w in words])
    sentences: list[Sentence] = []
    units: list[Unit] = []

    def emit(group: list[tuple[list[Word], list[str]]]) -> None:
        for ws, sids in _merge_tiny(group, p):
            spk = ws[0].speaker or "S1"
            units.append(
                Unit(
                    id=f"u{ws[0].start_ms:07d}-{spk}",
                    speaker=spk,
                    sentence_ids=sids,
                    start_ms=ws[0].start_ms,
                    end_ms=ws[-1].end_ms,
                    text=join_words(ws, p.joiner),
                    n_words=len(ws),
                    asr_conf=round(sum(w.conf for w in ws) / len(ws), 3),
                )
            )

    for block, bprobs in _blocks(words, probs):
        # A "group" holds the chunks between two mandatory cuts (long pause or speaker
        # change); tiny units may only merge inside their group.
        group: list[tuple[list[Word], list[str]]] = []
        prev_end: int | None = None
        for sw, sprobs in _sentences(block, bprobs, p.threshold):
            sid = f"s{sw[0].start_ms:07d}-{sw[0].speaker}"
            sentences.append(
                Sentence(
                    id=sid,
                    speaker=sw[0].speaker or "S1",
                    start_ms=sw[0].start_ms,
                    end_ms=sw[-1].end_ms,
                    text=join_words(sw, p.joiner),
                )
            )
            for pw, pp, after_pause in _pause_pieces(sw, sprobs, p):
                gap = pw[0].start_ms - prev_end if prev_end is not None else 0
                if group and (after_pause or gap >= p.hard_pause_ms):
                    emit(group)
                    group = []
                group.extend((c, [sid]) for c in split_long(pw, p, pp))
                prev_end = pw[-1].end_ms
        if group:
            emit(group)
    units.sort(key=lambda u: (u.start_ms, u.speaker))
    units = snap_to_speech(units, req.speech)
    return Segmentation(lang=req.lang, sentences=sentences, units=units)
