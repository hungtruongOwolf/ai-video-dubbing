"""Voice references: one clean 3-10 s clip per speaker, cut from the separated vocals.

Voice-cloning TTS copies the timbre of this clip, and some engines also want its transcript.
Choice per speaker: the unit with the highest ASR confidence whose length is within
[min_ms, max_ms], preferring lengths near 6 s. Without such a unit, the longest unit is used
(trimmed to max_ms, with its transcript trimmed proportionally).

M4 replaces this with a scored reference bank (anchor + local clips, speaker-similarity gate).
"""

from __future__ import annotations

from dub.contracts.v1 import RefBank, RefsReq, Unit, VoiceRef
from dub.ops.media import run_ffmpeg
from dub.store.cas import CAS

PAD_MS = 60  # keep a little air around the first and last word


def _pick(units: list[Unit], min_ms: int, max_ms: int) -> Unit:
    fitting = [u for u in units if min_ms <= u.slot_ms <= max_ms]
    if fitting:
        return max(fitting, key=lambda u: (round(u.asr_conf, 2), -abs(u.slot_ms - 6000)))
    return max(units, key=lambda u: u.slot_ms)


def extract_refs(req: RefsReq, cas: CAS) -> RefBank:
    by_speaker: dict[str, list[Unit]] = {}
    for u in req.units:
        by_speaker.setdefault(u.speaker, []).append(u)
    refs: list[VoiceRef] = []
    for speaker, units in sorted(by_speaker.items()):
        u = _pick(units, req.min_ms, req.max_ms)
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
                "-ss",
                f"{start / 1000:.3f}",
                "-to",
                f"{end / 1000:.3f}",
                "-i",
                str(cas.path(req.vocals_key)),
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                str(out),
            ]
        )
        refs.append(
            VoiceRef(
                speaker=speaker,
                audio_key=cas.put_file(out, move=True),
                text=text,
                seconds=round((end - start) / 1000, 2),
            )
        )
    return RefBank(refs=refs)
