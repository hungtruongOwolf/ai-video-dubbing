"""Voice references: one clean 3-10 s clip per speaker, cut from the separated vocals.

Voice-cloning TTS copies the timbre of this clip, and some engines also want its transcript.

With speaker embeddings of every unit (`unit_vectors`, from the audio.embed op), each speaker
gets a centroid, and the reference is the 3-10 s unit that is most typical of the speaker:
closest to the centroid, with only that speaker in it (purity >= 0.95) and a confident
transcript. That avoids clips with laughter, music or another voice in them. The centroid
and the speaker's own spread (`self_sim`) are kept for the QA gate's voice check.

With `candidates` > 1 the next most typical clips come back as alternates: the pipeline lets
each one say two test sentences and keeps the clip whose voice comes out closest to the
speaker (the audition, see pipeline._audition_voices). Being typical of the speaker in the
original is not the same as cloning well: noise, reverberation or a shout in the clip carry
over into every line.

Without embeddings (fakes, tests) the unit with the highest ASR confidence near 6 s is used.
Without a unit of the right length, the longest one is trimmed to max_ms.
"""

from __future__ import annotations

from dub.contracts.v1 import RefBank, RefsReq, Unit, VoiceRef
from dub.ops.media import run_ffmpeg
from dub.store.cas import CAS

PAD_MS = 60  # keep a little air around the first and last word


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


def _rank(
    units: list[Unit], min_ms: int, max_ms: int, sims: dict[str, float], purity: dict[str, float]
) -> list[Unit]:
    """Candidate clips of one speaker, best first."""
    fitting = [u for u in units if min_ms <= u.slot_ms <= max_ms]
    pure = [u for u in fitting if purity.get(u.id, 1.0) >= 0.95] or fitting
    if pure and sims:
        return sorted(pure, key=lambda u: (-round(sims.get(u.id, 0.0), 2), -round(u.asr_conf, 2)))
    if pure:
        return sorted(pure, key=lambda u: (-round(u.asr_conf, 2), abs(u.slot_ms - 6000)))
    return [max(units, key=lambda u: u.slot_ms)]


def extract_refs(req: RefsReq, cas: CAS) -> RefBank:
    by_speaker: dict[str, list[Unit]] = {}
    for u in req.units:
        by_speaker.setdefault(u.speaker, []).append(u)
    refs: list[VoiceRef] = []
    alternates: list[VoiceRef] = []
    for speaker, units in sorted(by_speaker.items()):
        vecs = {u.id: _unit_vec(req.unit_vectors[u.id]) for u in units if u.id in req.unit_vectors}
        centroid = _centroid(list(vecs.values())) if vecs else []
        sims = {uid: _cos(v, centroid) for uid, v in vecs.items()}
        self_sim = sum(sims.values()) / len(sims) if sims else 0.0
        ranked = _rank(units, req.min_ms, req.max_ms, sims, req.purity)
        for i, u in enumerate(ranked[: max(1, req.candidates)]):
            ref = _cut(req, cas, speaker, u)
            ref = ref.model_copy(
                update={
                    "centroid": [round(x, 5) for x in centroid],
                    "self_sim": round(self_sim, 4),
                    "sim": round(sims.get(u.id, 0.0), 4),
                }
            )
            (refs if i == 0 else alternates).append(ref)
    return RefBank(refs=refs, alternates=alternates)


def _cut(req: RefsReq, cas: CAS, speaker: str, u: Unit) -> VoiceRef:
    """The clip of one unit from the separated vocals (mono, 24 kHz), with its words."""
    start = max(0, u.start_ms - PAD_MS)
    end = min(u.end_ms + PAD_MS, start + req.max_ms)
    text = u.text
    if u.slot_ms > req.max_ms:  # trimmed: keep the matching share of the words
        words = text.split()
        keep = max(1, int(len(words) * req.max_ms / u.slot_ms))
        text = " ".join(words[:keep])
    out = cas.tmp_path(".wav")
    run_ffmpeg(
        [
            "-ss", f"{start / 1000:.3f}", "-to", f"{end / 1000:.3f}",
            "-i", str(cas.path(req.vocals_key)),
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(out),
        ]
    )  # fmt: skip
    return VoiceRef(
        speaker=speaker,
        audio_key=cas.put_file(out, move=True),
        text=text,
        seconds=round((end - start) / 1000, 2),
    )
