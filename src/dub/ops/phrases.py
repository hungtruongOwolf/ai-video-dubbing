"""Phrase timing: a dubbed line pauses where its actor pauses (M11).

A line is voiced as one take and fitted to its slot as a whole. The TTS pauses at its own
places (a comma, between two sentences) and speaks on through the actor's pauses. In the drama,
112 windows of 250 ms had the dub paused and the actor speaking, in lines whose level matched
the original's (M10 follow-up). Each mismatch is a mouth moving with no voice, or a voice over
a closed mouth.

This module re-times a take from the signal only, with no new translation and no new take:
1. The actor's phrases are the speech regions of the separated vocals inside the slot, joined
   across pauses shorter than `pause_ms`.
2. The take's cut points are its silences of at least `cut_ms`. A cut there never splits a
   syllable.
3. Each pause of the actor is matched to the take's silence at the same share of speech. The
   matching is monotonic (dynamic programming). A pause stays unmatched when no silence of the
   take is within `max_shift` of it.
4. The take is cut at the matched silences. Each piece starts where its phrase starts and is
   sped up or slowed down, within `rate_lo`..`rate_hi`, to end where the phrase ends.
5. A silence of the take that matched no pause and is longer than `long_pause_ms` is shortened
   to `keep_pause_ms`: the actor speaks through it.
The line keeps its onset (the timing fit placed it) and never ends later than `end_max_ms`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dub.contracts.v1 import PhraseReq, PhraseRes
from dub.ops.audio_io import read_wav, write_wav
from dub.ops.media import run_ffmpeg
from dub.store.cas import CAS

Span = tuple[int, int]


@dataclass(frozen=True)
class Piece:
    """[src0, src1) of the take, played at [out0, out1) of the re-timed clip."""

    src0: int
    src1: int
    out0: int
    out1: int

    @property
    def rate(self) -> float:  # > 1: faster
        return (self.src1 - self.src0) / max(1, self.out1 - self.out0)


@dataclass
class Plan:
    pieces: list[Piece]
    matched: int  # actor pauses the take now pauses at
    shortened: int  # pauses of the take the actor speaks through, shortened


def join(regions: list[Span], gap_ms: int) -> list[Span]:
    """Regions joined across gaps shorter than `gap_ms`."""
    out: list[Span] = []
    for s, e in sorted(regions):
        if out and s - out[-1][1] < gap_ms:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        elif e > s:
            out.append((s, e))
    return out


def _share_before(regions: list[Span], t: int) -> float:
    """Share of the speech in `regions` that comes before time t."""
    total = sum(e - s for s, e in regions) or 1
    return sum(max(0, min(e, t) - s) for s, e in regions) / total


def match(
    pauses: list[float], gaps: list[float], gap_len: list[int], *, max_shift: float
) -> list[tuple[int, int]]:
    """Monotonic matching of the actor's pauses to the take's silences, both given as the
    share of speech before them: (pause index, gap index) pairs. A match costs the difference
    of the shares, less a little for a long silence (a real break, not a stop consonant). An
    unmatched pause costs `max_shift`, so no match is worse than that."""
    k, m = len(pauses), len(gaps)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(k + 1)]
    back: list[list[tuple[int, int, bool] | None]] = [[None] * (m + 1) for _ in range(k + 1)]
    for j in range(m + 1):
        cost[0][j] = 0.0
    for i in range(1, k + 1):
        cost[i][0] = cost[i - 1][0] + max_shift
        back[i][0] = (i - 1, 0, False)
        for j in range(1, m + 1):
            best, how = cost[i][j - 1], (i, j - 1, False)  # silence j unused
            if cost[i - 1][j] + max_shift < best:  # pause i unmatched
                best, how = cost[i - 1][j] + max_shift, (i - 1, j, False)
            d = abs(pauses[i - 1] - gaps[j - 1])
            if d <= max_shift:
                c = cost[i - 1][j - 1] + d - (0.02 if gap_len[j - 1] >= 150 else 0.0)
                if c < best:
                    best, how = c, (i - 1, j - 1, True)
            cost[i][j], back[i][j] = best, how
    pairs: list[tuple[int, int]] = []
    i, j = k, m
    while i > 0 and back[i][j] is not None:
        pi, pj, took = back[i][j]  # type: ignore[misc]
        if took:
            pairs.append((i - 1, j - 1))
        i, j = pi, pj
    return pairs[::-1]


def plan(
    actor: list[Span],
    voiced: list[Span],
    *,
    end_max_ms: int,
    pause_ms: int = 250,
    cut_ms: int = 80,
    long_pause_ms: int = 220,
    keep_pause_ms: int = 100,
    max_shift: float = 0.15,
    rate_lo: float = 0.87,
    rate_hi: float = 1.15,
    min_gap_ms: int = 60,
) -> Plan | None:
    """The take's speech pieces and where they play, or None when nothing should change.

    `actor`: the actor's speech regions inside the slot; `voiced`: the take's speech regions
    (gaps of at least `cut_ms` between them); both in ms from the start of the take as placed."""
    voiced = join(voiced, cut_ms)
    if not voiced or not actor:
        return None
    blocks = join(actor, pause_ms)
    # the take's silences: (start, end), with the share of its speech before them
    gaps = [(voiced[i][1], voiced[i + 1][0]) for i in range(len(voiced) - 1)]
    a_pauses = [(blocks[i][1], blocks[i + 1][0]) for i in range(len(blocks) - 1)]
    pairs = match(
        [_share_before(actor, (a + b) // 2) for a, b in a_pauses],
        [_share_before(voiced, (a + b) // 2) for a, b in gaps],
        [b - a for a, b in gaps],
        max_shift=max_shift,
    )
    by_gap = {j: i for i, j in pairs}
    shorten = {j for j, (a, b) in enumerate(gaps) if j not in by_gap and b - a > long_pause_ms}
    if not pairs and not shorten:
        return None
    # cut the take at the matched and the shortened silences: groups of pieces, one group per
    # stretch of the actor between two matched pauses
    groups: list[list[Span]] = [[]]
    start = voiced[0][0]
    for j, (a, b) in enumerate(gaps):
        if j in by_gap or j in shorten:
            groups[-1].append((start, a))
            start = b
        if j in by_gap:
            groups.append([])
    groups[-1].append((start, voiced[-1][1]))
    # where each group plays: the line keeps its onset and its end; inside, the stretches of
    # the actor between the matched pauses
    edges = [voiced[0][0]]
    for i, _j in pairs:
        edges += [a_pauses[i][0], a_pauses[i][1]]
    edges.append(max(voiced[-1][1], min(blocks[-1][1], end_max_ms)))
    targets = [(edges[2 * g], edges[2 * g + 1]) for g in range(len(groups))]

    pieces: list[Piece] = []
    prev_end = None
    for group, (w0, w1) in zip(groups, targets, strict=True):
        speech = sum(e - s for s, e in group)
        pauses = keep_pause_ms * (len(group) - 1)
        t0 = w0 if prev_end is None else max(w0, prev_end + min_gap_ms)
        room = max(1, w1 - t0 - pauses)
        rate = min(max(speech / room, rate_lo), rate_hi)
        t = t0
        for k, (s, e) in enumerate(group):
            if k:
                t += keep_pause_ms
            out = round((e - s) / rate)
            pieces.append(Piece(s, e, t, t + out))
            t += out
        prev_end = t
    if pieces[-1].out1 > end_max_ms:
        return None
    if all(abs(p.out0 - p.src0) < 20 and abs(p.out1 - p.src1) < 20 for p in pieces):
        return None
    return Plan(pieces, matched=len(pairs), shortened=len(shorten))


def take_regions(audio: np.ndarray, sr: int, *, frame_ms: int = 10, below_db: float = 35.0,
                 min_gap_ms: int = 80) -> list[Span]:  # fmt: skip
    """Speech regions of a clean TTS take: frames within `below_db` of its loud frames (95th
    percentile), with silences shorter than `min_gap_ms` (stop consonants) filled in."""
    hop = max(1, sr * frame_ms // 1000)
    n = len(audio) // hop
    if n == 0:
        return []
    frames = audio[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    thr = max(np.percentile(db, 95) - below_db, -60.0)
    out: list[Span] = []
    start = None
    for i, on in enumerate(db > thr):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start * frame_ms, i * frame_ms))
            start = None
    if start is not None:
        out.append((start * frame_ms, n * frame_ms))
    return join(out, min_gap_ms)


def warp(pieces: list[tuple[int, int, int, int]], t: float) -> float:
    """Where time t of the take plays in the re-timed clip (the back-check's words):
    `pieces` as (src0, src1, out0, out1)."""
    for k, (s0, s1, o0, o1) in enumerate(pieces):
        if t <= s1:
            if t < s0:  # in a silence before this piece
                return o0 - (s0 - t) if k == 0 else o0
            return o0 + (t - s0) * (o1 - o0) / max(1, s1 - s0)
    s0, s1, o0, o1 = pieces[-1]
    return o1 + (t - s1)


def _stretched(piece: np.ndarray, sr: int, rate: float, engine: str, cas: CAS) -> np.ndarray:
    if abs(rate - 1.0) < 0.02 or len(piece) < sr // 20:
        return piece
    src, out = cas.tmp_path(".wav"), cas.tmp_path(".wav")
    write_wav(src, piece, sr)
    filt = (f"rubberband=tempo={rate:.5f}:pitchq=quality:transients=smooth"
            if engine == "rubberband" else f"atempo={rate:.5f}")  # fmt: skip
    run_ffmpeg(["-i", str(src), "-af", filt, "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
                str(out)])  # fmt: skip
    audio, _ = read_wav(out)
    src.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return audio.mean(axis=1)


def align(req: PhraseReq, cas: CAS) -> PhraseRes:
    """The take re-timed to its actor's phrases (see the module docstring)."""
    audio, sr = read_wav(cas.path(req.audio_key))
    mono = audio.mean(axis=1)
    dur = round(len(mono) * 1000 / sr)
    p = plan(
        [tuple(r) for r in req.actor], take_regions(mono, sr, min_gap_ms=req.cut_ms),
        end_max_ms=req.end_max_ms, pause_ms=req.pause_ms, cut_ms=req.cut_ms,
        rate_lo=req.rate_lo, rate_hi=req.rate_hi,
    )  # fmt: skip
    if p is None:
        return PhraseRes(audio_key=req.audio_key, dur_ms=dur)
    fade = sr * 5 // 1000
    tail = max(0, dur - p.pieces[-1].src1)  # the air after the voice stays
    out = np.zeros(int((p.pieces[-1].out1 + 1.2 * tail) * sr / 1000) + sr // 50, dtype=np.float32)
    used = 0
    for pc in p.pieces:
        a, b = pc.src0 * sr // 1000, pc.src1 * sr // 1000
        if pc is p.pieces[0]:
            a = 0  # the air before the voice stays too
        if pc is p.pieces[-1]:
            b = len(mono)
        seg = _stretched(mono[a:b].copy(), sr, pc.rate, req.engine, cas)
        if len(seg) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        o = int((pc.out0 - (pc.src0 - a * 1000 // sr) / pc.rate) * sr / 1000)
        o = max(0, o)
        n = min(len(seg), len(out) - o)
        out[o : o + n] += seg[:n]
        used = max(used, o + n)
    out = out[:used]
    path = cas.tmp_path(".wav")
    write_wav(path, out, sr)
    key = cas.put_file(path, move=True)
    return PhraseRes(
        audio_key=key, dur_ms=round(len(out) * 1000 / sr), matched=p.matched,
        shortened=p.shortened, pieces=[(pc.src0, pc.src1, pc.out0, pc.out1) for pc in p.pieces],
    )  # fmt: skip
