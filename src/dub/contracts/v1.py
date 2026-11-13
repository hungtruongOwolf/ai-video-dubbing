"""Data contracts (/v1) passed between pipeline steps.

Every step (an "op") takes a `...Req` object and returns a result object. These objects:
- are pydantic models, so they are type-checked and can be exported as JSON Schema
  (`dub schema`);
- only carry small data (text, numbers, timings). Large data such as audio lives in the
  CAS and is referenced by a "CAS key" (a string like `3fa2...9c.wav`).

The same contract therefore works for in-process providers, CLI providers and (from M1)
workers running on the GPU pod behind HTTP.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "v1"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- media


class DecodeReq(Model):
    src_key: str  # original audio track extracted from the video (.mka, not decoded)


class MediaInfo(Model):
    src_key: str
    duration_ms: int
    a48k_key: str  # 48 kHz stereo WAV, used for mixing
    a16k_key: str  # 16 kHz mono WAV, used for ASR and VAD


# ---------------------------------------------------------------- separation


class SeparateReq(Model):
    audio_key: str  # 48 kHz stereo WAV


class Stems(Model):
    vocals_key: str
    instrumental_key: str  # the "bed": music + effects, no speech


# ---------------------------------------------------------------- ASR


class Word(Model):
    w: str  # one word (or one character for zh/ja); may carry punctuation: "everyone,"
    start_ms: int
    end_ms: int
    conf: float = 1.0
    speaker: str | None = None


class TranscribeReq(Model):
    audio_key: str
    lang: str = "auto"
    replay_key: str | None = None  # fake provider: replay an existing transcript


class WordTimeline(Model):
    lang: str
    words: list[Word]


# ---------------------------------------------------------------- diarization

Gender = Literal["female", "male", "unknown"]
AgeBand = Literal["child", "young", "adult", "senior", "unknown"]


class DiarizeReq(Model):
    audio_key: str
    speakers_hint: int | None = None
    replay_key: str | None = None


class Turn(Model):
    speaker: str
    start_ms: int
    end_ms: int


class SpeakerVoice(Model):
    """Guess made from the voice. It is ONE piece of evidence, never the truth."""

    speaker: str
    gender: Gender = "unknown"
    gender_conf: float = 0.0
    age_band: AgeBand = "unknown"


class Diarization(Model):
    turns: list[Turn]
    speakers: list[str]
    voices: list[SpeakerVoice] = Field(default_factory=list)


# ---------------------------------------------------------------- segmentation


class SegmentParams(Model):
    max_unit_ms: int = 15000  # a dub unit is never longer than this
    target_unit_ms: int = 7000  # the "nice" length the splitter aims for
    min_unit_ms: int = 1000  # shorter units get merged when possible
    hard_pause_ms: int = 1500  # a longer silence is a mandatory boundary
    joiner: str = " "  # "" for zh/ja/th


class SegmentReq(Model):
    lang: str
    words: list[Word]
    turns: list[Turn]
    params: SegmentParams


class Sentence(Model):
    id: str
    speaker: str
    start_ms: int
    end_ms: int
    text: str


class Unit(Model):
    """Dub unit: one speaker, 1-15 s. The unit we translate, voice, time-fit and edit."""

    id: str
    speaker: str
    sentence_ids: list[str]
    start_ms: int
    end_ms: int
    text: str
    n_words: int
    asr_conf: float = 1.0

    @property
    def slot_ms(self) -> int:
        return self.end_ms - self.start_ms


class Segmentation(Model):
    lang: str
    sentences: list[Sentence]
    units: list[Unit]


# ---------------------------------------------------------------- cast sheet (forms of address)

Formality = Literal["formal", "polite", "casual", "intimate", "unknown"]


class CastSpeaker(Model):
    id: str
    name: str | None = None
    gender: Gender = "unknown"
    conf: float = 0.0
    age_band: AgeBand = "unknown"
    role: str | None = None
    style: str | None = None
    evidence: list[str] = Field(default_factory=list)
    locked: bool = False  # True = confirmed by the user; the LLM must not change it


class AddressTerms(Model):
    self_term: str | None = None  # how the speaker refers to themself, e.g. vi "cháu"
    other_term: str | None = None  # how the speaker addresses the listener, e.g. vi "chú"


class Relation(Model):
    speaker: str
    listener: str
    relation: str | None = None  # short description: "host -> older guest"
    formality: Formality = "unknown"
    since_ms: int = 0  # relations can change during a video
    address: dict[str, AddressTerms] = Field(default_factory=dict)  # keyed by target language
    conf: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class Audience(Model):
    formality: Formality = "unknown"
    address: dict[str, str] = Field(default_factory=dict)  # e.g. {"vi": "các bạn"}


class Mentioned(Model):
    name: str
    gender: Gender = "unknown"
    note: str | None = None


class CastSheet(Model):
    speakers: list[CastSpeaker] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    audience: Audience = Field(default_factory=Audience)
    mentioned: list[Mentioned] = Field(default_factory=list)

    def rows_for(self, speaker_ids: set[str]) -> CastSheet:
        """Keep only the rows about speakers present in one translation window.

        A window's cache key contains only this subset, so editing S2 re-translates
        only the windows where S2 appears.
        """
        return CastSheet(
            speakers=[s for s in self.speakers if s.id in speaker_ids],
            relations=[
                r for r in self.relations if r.speaker in speaker_ids or r.listener in speaker_ids
            ],
            audience=self.audience,
            mentioned=self.mentioned,
        )


# ---------------------------------------------------------------- translation


class ContextReq(Model):
    src_lang: str
    tgt_lang: str
    units: list[Unit]
    voices: list[SpeakerVoice] = Field(default_factory=list)


class GlossaryEntry(Model):
    source: str
    target: str


class ContextSheet(Model):
    summary: str = ""
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    cast: CastSheet = Field(default_factory=CastSheet)
    flags: list[str] = Field(default_factory=list)  # uncertain points a human should confirm

    def subset_for(self, units: list[Unit]) -> ContextSheet:
        speakers = {u.speaker for u in units}
        text = " ".join(u.text.lower() for u in units)
        return ContextSheet(
            summary=self.summary,
            glossary=[g for g in self.glossary if g.source.lower() in text],
            cast=self.cast.rows_for(speakers),
            flags=[],
        )


class TranslateReq(Model):
    src_lang: str
    tgt_lang: str
    units: list[Unit]  # units to translate in this window
    before: list[Unit] = Field(default_factory=list)  # context only, not translated
    after: list[Unit] = Field(default_factory=list)
    context: ContextSheet
    speech_unit: Literal["syllable", "char", "word"] = "syllable"
    units_per_s: float = 4.5  # estimated speaking rate of the target language


class Translation(Model):
    unit_id: str
    short: str
    target: str
    long: str
    listener: str | None = None  # who this line is spoken to (S1, S2, audience)
    self_term: str | None = None
    other_term: str | None = None
    note: str | None = None


class TranslationBatch(Model):
    items: list[Translation]


class RewriteReq(Model):
    unit_id: str
    tgt_lang: str
    source_text: str
    text: str
    target_units: int  # syllables/characters the line should have
    current_units: int


class RewriteRes(Model):
    text: str


# ---------------------------------------------------------------- TTS


class SynthReq(Model):
    unit_id: str
    text: str
    lang: str
    speaker: str
    voice_key: str | None = None  # 3-10 s reference clip (from M4)
    seed: int = 0
    target_ms: int | None = None  # only used when caps.duration_control == "exact"


class SynthRes(Model):
    unit_id: str
    audio_key: str
    sample_rate: int
    dur_ms: int
    provider: str = ""
    exact: bool = False  # True if the engine honoured target_ms


# ---------------------------------------------------------------- timing, mix, subs


class Placement(Model):
    unit_id: str
    speaker: str
    slot_start_ms: int
    slot_end_ms: int
    start_ms: int
    end_ms: int
    natural_ms: int  # natural spoken length (predicted or measured)
    clip_ms: int  # length of the clip the TTS returned
    rate: float  # total speed-up = natural / (end - start)
    variant: str  # short | target | long | rewrite
    text: str
    rewrites: int = 0
    flags: list[str] = Field(default_factory=list)
    audio_key: str | None = None  # time-stretched clip, ready to mix


class StretchReq(Model):
    audio_key: str
    rate: float
    sample_rate: int = 48000


class AudioRes(Model):
    audio_key: str
    dur_ms: int


class MixReq(Model):
    bed_key: str
    duration_ms: int
    placements: list[Placement]
    bed_gain_db: float = 0.0
    loudness_lufs: float = -16.0


class Cue(Model):
    start_ms: int
    end_ms: int
    text: str


class SubsReq(Model):
    lang: str
    items: list[Cue]  # one item per line with timing; gets wrapped/split into cues
    cpl: int = 42
    cps: float = 17.0
    max_lines: int = 2
    joiner: str = " "


class SubtitleTrack(Model):
    lang: str
    cues: list[Cue]
    srt_key: str


# ---------------------------------------------------------------- routing trace


class Rejection(Model):
    provider: str
    reason: str


class RouteTrace(Model):
    op: str
    subject: str | None = None  # unit id, window id... when routing per item
    source: Literal["override", "rule", "langpack", "default"]
    rule_id: str | None = None
    chain: list[str]
    rejected: list[Rejection] = Field(default_factory=list)


ALL_MODELS: list[type[BaseModel]] = [
    DecodeReq,
    MediaInfo,
    SeparateReq,
    Stems,
    Word,
    TranscribeReq,
    WordTimeline,
    DiarizeReq,
    Turn,
    SpeakerVoice,
    Diarization,
    SegmentParams,
    SegmentReq,
    Sentence,
    Unit,
    Segmentation,
    CastSpeaker,
    AddressTerms,
    Relation,
    Audience,
    Mentioned,
    CastSheet,
    ContextReq,
    GlossaryEntry,
    ContextSheet,
    TranslateReq,
    Translation,
    TranslationBatch,
    RewriteReq,
    RewriteRes,
    SynthReq,
    SynthRes,
    Placement,
    StretchReq,
    AudioRes,
    MixReq,
    Cue,
    SubsReq,
    SubtitleTrack,
    Rejection,
    RouteTrace,
]
