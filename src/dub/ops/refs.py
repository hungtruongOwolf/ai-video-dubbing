"""Voice references: one clean 3-10 s clip per speaker, cut from the separated vocals.

Voice-cloning TTS copies the timbre of this clip, and OmniVoice also wants its transcript.

With speaker embeddings of every unit (`unit_vectors`, from the audio.embed op), each speaker
gets a centroid, and the reference is the 3-10 s unit that is most typical of the speaker:
closest to the centroid, with only that speaker in it (purity >= 0.95) and a confident
transcript. That avoids clips with laughter, music or another voice in them. The centroid
and the speaker's own spread (`self_sim`) are kept for the QA gate's voice check.

Clean edges matter as much as the clip (M8, from VoiceStudio's notes and our zh -> vi drama):
OmniVoice continues the reference, so a clip that starts or ends inside a word, or whose
transcript does not match its audio, makes the clone speak a fragment of the reference
language in the dub. So:
- a unit within 300 ms of another speaker's line ranks last (its edges may hold their voice);
- each edge is moved to the quietest 20 ms frame near the unit boundary (searching 300 ms
  outward and 100 ms inward), never inside the speech;
- the pipeline transcribes the cut clip again (pipeline._retranscribe_refs), so the text is
  what the clip says, not what the unit said.
A speaker with no 3-10 s unit (a minor character in a drama) gets several of their own clean
lines of at least 0.8 s joined with 20 ms of silence, up to 6 s, instead of one short clip.

With `candidates` > 1 the next most typical clips come back as alternates (the audition).

Without embeddings (fakes, tests) the unit with the highest ASR confidence near 6 s is used.
"""

from __future__ import annotations

import numpy as np

from dub.contracts.v1 import RefBank, RefsReq, Unit, VoiceRef
from dub.ops.audio_io import read_wav_range, resample_int, write_wav
from dub.store.cas import CAS

SR_IN = 48000  # separated vocals
SR_OUT = 24000  # OmniVoice's rate
FRAME_MS = 20
EDGE_OUT_MS = 300  # look this far outside the unit for a quiet cut point
EDGE_IN_MS = 100  # and this far inside it
NEIGHBOUR_MS = 300  # another speaker this close: the unit ranks last
POOL_MIN_MS = 800  # pooled lines are at least this long
POOL_TARGET_MS = 6000
POOL_GAP_MS = 20


def _unit_vec(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _centroid(vecs: list[list[float]]) -> list[float]:
    return _unit_vec([sum(col) / len(vecs) for col in zip(*vecs, strict=True)])


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def cos_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two embeddings (any length scale)."""
    return _cos(_unit_vec(a), _unit_vec(b))


def _crowded(units: list[Unit]) -> set[str]:
    """Units with another speaker's line within NEIGHBOUR_MS of either edge."""
    order = sorted(units, key=lambda u: u.start_ms)
    out = set()
    for i, u in enumerate(order):
        for j in (i - 1, i + 1):
            if 0 <= j < len(order) and order[j].speaker != u.speaker:
                v = order[j]
                gap = v.start_ms - u.end_ms if j > i else u.start_ms - v.end_ms
                if gap < NEIGHBOUR_MS:
                    out.add(u.id)
    return out


def _rank(
    units: list[Unit],
    min_ms: int,
    max_ms: int,
    sims: dict[str, float],
    purity: dict[str, float],
    crowded: set[str] | None = None,
) -> list[Unit]:
    """Candidate clips of one speaker (3-10 s), best first; [] when there is none."""
    crowded = crowded or set()
    fitting = [u for u in units if min_ms <= u.slot_ms <= max_ms]
    pure = [u for u in fitting if purity.get(u.id, 1.0) >= 0.95] or fitting
    if sims:
        return sorted(pure, key=lambda u: (u.id in crowded, -round(sims.get(u.id, 0.0), 2),
                                           -round(u.asr_conf, 2)))  # fmt: skip
    return sorted(pure, key=lambda u: (u.id in crowded, -round(u.asr_conf, 2),
                                       abs(u.slot_ms - 6000)))  # fmt: skip


def _quiet_edge(power: np.ndarray, lo: int, hi: int, prefer: int) -> int:
    """Index of the quietest frame in [lo, hi), the one nearest `prefer` on ties."""
    lo, hi = max(0, lo), min(len(power), hi)
    if hi <= lo:
        return min(max(prefer, 0), len(power))
    seg = power[lo:hi]
    best = np.flatnonzero(seg <= seg.min() * 1.0001) + lo
    return int(best[np.argmin(np.abs(best - prefer))])


def _clip(vocals, start_ms: int, end_ms: int, max_ms: int) -> np.ndarray:
    """The unit's audio (mono, 48 kHz) with both edges moved to quiet frames."""
    a = max(0, start_ms - EDGE_OUT_MS)
    audio, sr = read_wav_range(
        vocals, int(a * SR_IN / 1000), int((end_ms + EDGE_OUT_MS - a) * SR_IN / 1000)
    )
    if sr != SR_IN:
        raise ValueError(f"The vocals must be {SR_IN} Hz, got {sr}")
    mono = audio.mean(axis=1)
    n = int(SR_IN * FRAME_MS / 1000)
    k = len(mono) // n
    if k < 3:
        return mono
    power = (mono[: k * n].astype(np.float64).reshape(k, n) ** 2).mean(axis=1)
    f = lambda ms: (ms - a) // FRAME_MS  # noqa: E731 - frame index of a time
    s = _quiet_edge(power, f(start_ms - EDGE_OUT_MS), f(start_ms + EDGE_IN_MS), f(start_ms))
    e_ms = min(end_ms, start_ms + max_ms)
    e = _quiet_edge(power, f(e_ms - EDGE_IN_MS), f(e_ms + EDGE_OUT_MS) + 1, f(e_ms)) + 1
    return mono[s * n : max(e, s + 1) * n]


def _write(cas: CAS, mono48: np.ndarray, speaker: str, text: str) -> VoiceRef:
    out = cas.tmp_path(".wav")
    write_wav(out, resample_int(mono48, 1, 2), SR_OUT)
    return VoiceRef(
        speaker=speaker,
        audio_key=cas.put_file(out, move=True),
        text=text,
        seconds=round(len(mono48) / SR_IN, 2),
    )


def _pooled(req: RefsReq, cas: CAS, speaker: str, units: list[Unit], key) -> VoiceRef:
    """Several clean lines of a speaker who has no 3-10 s line, joined."""
    vocals = cas.path(req.vocals_key)
    pool = sorted((u for u in units if u.slot_ms >= POOL_MIN_MS), key=key) or [
        max(units, key=lambda u: u.slot_ms)
    ]
    parts: list[np.ndarray] = []
    texts: list[str] = []
    total = 0
    gap = np.zeros(int(SR_IN * POOL_GAP_MS / 1000), dtype=np.float32)
    for u in pool:
        piece = _clip(vocals, u.start_ms, u.end_ms, req.max_ms)
        if parts and total + len(piece) * 1000 // SR_IN > req.max_ms:
            break
        parts += [gap, piece] if parts else [piece]
        texts.append(u.text)
        total += len(piece) * 1000 // SR_IN
        if total >= POOL_TARGET_MS:
            break
    return _write(cas, np.concatenate(parts), speaker, " ".join(texts))


def extract_refs(req: RefsReq, cas: CAS) -> RefBank:
    by_speaker: dict[str, list[Unit]] = {}
    for u in req.units:
        by_speaker.setdefault(u.speaker, []).append(u)
    crowded = _crowded(req.units)
    refs: list[VoiceRef] = []
    alternates: list[VoiceRef] = []
    for speaker, units in sorted(by_speaker.items()):
        vecs = {u.id: _unit_vec(req.unit_vectors[u.id]) for u in units if u.id in req.unit_vectors}
        centroid = _centroid(list(vecs.values())) if vecs else []
        sims = {uid: _cos(v, centroid) for uid, v in vecs.items()}
        self_sim = sum(sims.values()) / len(sims) if sims else 0.0
        extra = {
            "centroid": [round(x, 5) for x in centroid],
            "self_sim": round(self_sim, 4),
        }
        ranked = _rank(units, req.min_ms, req.max_ms, sims, req.purity, crowded)
        if not ranked:
            key = lambda u, s=sims: (u.id in crowded, req.purity.get(u.id, 1.0) < 0.95,  # noqa: E731
                                     -round(s.get(u.id, 0.0), 2), -u.slot_ms)  # fmt: skip
            refs.append(_pooled(req, cas, speaker, units, key).model_copy(update=extra))
            continue
        vocals = cas.path(req.vocals_key)
        for i, u in enumerate(ranked[: max(1, req.candidates)]):
            ref = _write(cas, _clip(vocals, u.start_ms, u.end_ms, req.max_ms), speaker, u.text)
            ref = ref.model_copy(update={**extra, "sim": round(sims.get(u.id, 0.0), 4)})
            (refs if i == 0 else alternates).append(ref)
    return RefBank(refs=refs, alternates=alternates)
