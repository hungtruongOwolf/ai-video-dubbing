"""Fake providers: let the whole pipeline run on a laptop without any AI model.

They are deterministic (same input -> same output), fast, and produce real files so the
output video plays. Use them with `dub run ... --fake`.

- fake_separate: "bed" = original mix at -20 dB (a real separator removes the speech).
- fake_asr: replays a transcript (`<video>.words.json`) if given, otherwise invents
  placeholder words over the speech regions found by an energy VAD. As a back-check of a
  dubbed clip it "hears" exactly the expected text, spread over the voice.
- fake_diarize: speakers from the replayed transcript, otherwise a single speaker S1.
- fake_mt: "translation" = the source text with a "[vi]" tag; short/long variants are
  shorter/longer so the timing ladder has real choices to make.
- fake_tts: a buzzy synthetic voice whose pitch depends on the speaker.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from dub.contracts.v1 import (
    Attribution,
    CastSheet,
    CastSpeaker,
    Character,
    ContextReq,
    ContextSheet,
    Diarization,
    DiarizeReq,
    Embeddings,
    EmbedReq,
    RewriteReq,
    RewriteRes,
    SeparateReq,
    SpeakersReq,
    SpeakerVoice,
    Stems,
    SynthReq,
    SynthRes,
    TranscribeReq,
    TranslateReq,
    Translation,
    TranslationBatch,
    Turn,
    Word,
    WordTimeline,
)
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.audio_io import read_wav, write_wav
from dub.ops.synth import voice
from dub.ops.timing import count_units
from dub.ops.vad import energy_vad

LOREM = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua"
).split()


def stable_int(*parts: object) -> int:
    """Deterministic integer from any values (Python's hash() changes between runs)."""
    h = hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def load_replay(ctx: RunCtx, key: str) -> tuple[WordTimeline, list[SpeakerVoice]]:
    """A replay transcript is {"lang", "words": [...], "voices": [...] (optional)}."""
    data = json.loads(ctx.cas.read_text(key))
    tl = WordTimeline(lang=data["lang"], words=[Word.model_validate(w) for w in data["words"]])
    voices = [SpeakerVoice.model_validate(v) for v in data.get("voices", [])]
    return tl, voices


class _Base:
    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest

    def run(self, op, req, ctx: RunCtx):
        fn = getattr(self, "op_" + op.replace(".", "_"), None)
        if fn is None:
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        return fn(req, ctx)


# ---------------------------------------------------------------- separation


class FakeSeparator(_Base):
    def op_audio_separate(self, req: SeparateReq, ctx: RunCtx) -> Stems:
        audio, sr = read_wav(ctx.cas.path(req.audio_key))
        gain = 10 ** (float(ctx.params.get("bed_gain_db", -20.0)) / 20)
        tmp = ctx.cas.tmp_path(".wav")
        write_wav(tmp, audio * gain, sr)
        return Stems(vocals_key=req.audio_key, instrumental_key=ctx.cas.put_file(tmp, move=True))


# ---------------------------------------------------------------- ASR


class FakeASR(_Base):
    def op_asr_transcribe(self, req: TranscribeReq, ctx: RunCtx) -> WordTimeline:
        if req.replay_key:
            tl, _ = load_replay(ctx, req.replay_key)
            words = [
                w.model_copy(update={"speaker": None}) for w in tl.words
            ]  # ASR has no speakers
            return WordTimeline(lang=tl.lang if req.lang == "auto" else req.lang, words=words)
        if req.detect_only:  # a fake cannot tell languages apart
            return WordTimeline(lang="und", words=[], lang_prob=0.0)

        audio, sr = read_wav(ctx.cas.path(req.audio_key))
        words: list[Word] = []
        k = 0
        for start, end in energy_vad(audio.mean(axis=1), sr):
            n = max(1, round((end - start) / 1000 * 2.5))  # ~2.5 words per second
            step = (end - start) / n
            for j in range(n):
                text = LOREM[k % len(LOREM)]
                k += 1
                if j == n - 1:
                    text += "."
                elif j % 6 == 5:
                    text += ","
                ws = round(start + j * step)
                words.append(Word(w=text, start_ms=ws, end_ms=round(ws + step * 0.85), conf=0.9))
        return WordTimeline(lang="en" if req.lang == "auto" else req.lang, words=words)

    def op_asr_backcheck(self, req: TranscribeReq, ctx: RunCtx) -> WordTimeline:
        audio, sr = read_wav(ctx.cas.path(req.audio_key))
        lang = req.lang if req.lang != "auto" else "en"
        tokens = (req.expected_text or "").split()
        if not tokens or len(audio) == 0:
            return WordTimeline(lang=lang, words=[])
        start, end = 0, round(len(audio) * 1000 / sr)  # dubbed clips are trimmed to the voice
        weights = [max(1, count_units(t, "syllable")) for t in tokens]
        total, t = sum(weights), float(start)
        words = []
        for tok, w in zip(tokens, weights, strict=True):
            d = (end - start) * w / total
            words.append(Word(w=tok, start_ms=round(t), end_ms=round(t + d * 0.9), conf=0.9))
            t += d
        return WordTimeline(lang=lang, words=words)


# ---------------------------------------------------------------- diarization


class FakeDiarizer(_Base):
    def op_audio_embed(self, req: EmbedReq, ctx: RunCtx) -> Embeddings:
        """A crude 'voice print': pitch-like statistics per span (the fake voices differ by f0)."""
        audio, sr = read_wav(ctx.cas.path(req.audio_key))
        mono = audio.mean(axis=1)
        spans = req.spans or [(0, round(len(mono) * 1000 / sr))]
        out = []
        for s, e in spans:
            x = mono[s * sr // 1000 : max(e * sr // 1000, s * sr // 1000 + 2)]
            zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) if len(x) > 1 else 0.0
            rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
            v = np.array([zcr * 10, rms * 10, 1.0])
            v = v / (np.linalg.norm(v) + 1e-9)
            out.append([round(float(t), 4) for t in v])
        return Embeddings(vectors=out)

    def op_audio_diarize(self, req: DiarizeReq, ctx: RunCtx) -> Diarization:
        if req.replay_key:
            tl, voices = load_replay(ctx, req.replay_key)
            turns: list[Turn] = []
            for w in tl.words:
                spk = w.speaker or "S1"
                if turns and turns[-1].speaker == spk and w.start_ms - turns[-1].end_ms < 1500:
                    turns[-1].end_ms = w.end_ms
                else:
                    turns.append(Turn(speaker=spk, start_ms=w.start_ms, end_ms=w.end_ms))
            speakers = sorted({t.speaker for t in turns})
            return Diarization(turns=turns, speakers=speakers, voices=voices)

        audio, sr = read_wav(ctx.cas.path(req.audio_key))
        regions = energy_vad(audio.mean(axis=1), sr)
        turns = [Turn(speaker="S1", start_ms=s, end_ms=e) for s, e in regions]
        return Diarization(turns=turns, speakers=["S1"] if turns else [])


# ---------------------------------------------------------------- MT


def _cut_words(text: str, ratio: float) -> str:
    words = text.split()
    keep = max(1, int(round(len(words) * ratio)))
    return " ".join(words[:keep])


class FakeMT(_Base):
    def op_mt_context(self, req: ContextReq, ctx: RunCtx) -> ContextSheet:
        order: list[str] = []
        for u in req.units:
            if u.speaker not in order:
                order.append(u.speaker)
        voices = {v.speaker: v for v in req.voices}
        speakers = []
        for s in order:
            v = voices.get(s)
            speakers.append(
                CastSpeaker(
                    id=s,
                    gender=v.gender if v else "unknown",
                    conf=v.gender_conf if v else 0.0,
                    age_band=v.age_band if v else "unknown",
                    evidence=[f"voice hint: {v.gender} {v.gender_conf:.2f}"] if v else [],
                )
            )
        flags = [f"{s.id}: gender not confirmed by dialogue" for s in speakers]
        return ContextSheet(
            summary=f"(fake) {len(req.units)} units, {len(order)} speakers",
            cast=CastSheet(speakers=speakers),
            flags=flags,
        )

    def op_mt_translate(self, req: TranslateReq, ctx: RunCtx) -> TranslationBatch:
        items = []
        for u in req.units:
            target = f"[{req.tgt_lang}] {u.text}"
            extra = " ".join(u.text.split()[: max(1, len(u.text.split()) // 4)])
            items.append(
                Translation(
                    unit_id=u.id,
                    short=_cut_words(target, 0.75),
                    target=target,
                    long=f"{target} {extra}",
                    note="fake translation",
                )
            )
        return TranslationBatch(items=items)

    def op_mt_rewrite(self, req: RewriteReq, ctx: RunCtx) -> RewriteRes:
        ratio = req.target_units / max(req.current_units, 1)
        return RewriteRes(text=_cut_words(req.text, min(1.0, ratio)))

    def op_text_speakers(self, req: SpeakersReq, ctx: RunCtx) -> Attribution:
        """One character per diarizer voice: the voices' turns come back unchanged."""
        ids: dict[str, str] = {}
        for ln in req.lines:
            ids.setdefault(ln.voice, f"P{len(ids) + 1}")
        return Attribution(
            characters=[Character(id=c, who=f"voice {v}") for v, c in ids.items()],
            speaker=[ids[ln.voice] for ln in req.lines],
            sure=[True] * len(req.lines),
        )


# ---------------------------------------------------------------- TTS


class FakeTTS(_Base):
    def run_batch(self, op, reqs, ctx: RunCtx) -> list:
        """Batched calls (worker continuous batching): same results as one by one."""
        return [self.run(op, r, ctx) for r in reqs]

    def op_tts_synthesize(self, req: SynthReq, ctx: RunCtx) -> SynthRes:
        sr = 48000
        if req.target_ms:
            dur = int(req.target_ms)
        else:
            # A natural-length engine: ~5 syllables/s (~5.5 characters/s for CJK/Thai),
            # with a deterministic +-10% wobble so the timing ladder has work to do.
            cjk = req.lang.split("-")[0] in {"zh", "ja", "ko", "th"}
            units = count_units(req.text, "char" if cjk else "syllable")
            jitter = 0.9 + (stable_int(req.unit_id, req.seed, req.text) % 1000) / 1000 * 0.2
            dur = int(units / (5.5 if cjk else 5.0) * 1000 * jitter)
        f0 = 110 + (stable_int(req.speaker) % 8) * 18
        audio = voice(dur, f0, seed=stable_int(req.unit_id, req.seed) % (2**32), sr=sr)
        tmp = ctx.cas.tmp_path(".wav")
        write_wav(tmp, audio, sr)
        return SynthRes(
            unit_id=req.unit_id,
            audio_key=ctx.cas.put_file(tmp, move=True),
            sample_rate=sr,
            dur_ms=round(len(audio) * 1000 / sr),
            provider=self.manifest.id,
            exact=req.target_ms is not None,
        )


class FakeVoiceProfile(_Base):
    def op_audio_voice_profile(self, req, ctx: RunCtx):
        from dub.contracts.v1 import SpeakerVoice, VoiceProfiles

        return VoiceProfiles(voices=[SpeakerVoice(speaker=s) for s in sorted(set(req.speakers))])
