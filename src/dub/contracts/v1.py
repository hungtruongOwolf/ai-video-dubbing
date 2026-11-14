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

from typing import Any, Literal

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
    space_before: bool = False  # zh/ja/th...: the ASR put a space before this piece


def join_words(words: list[Word], joiner: str) -> str:
    """Text of consecutive words. Scripts without spaces keep the spaces the ASR produced
    (Thai separates phrases and sentences with them)."""
    if joiner:
        return joiner.join(w.w.strip() for w in words).strip()
    return "".join((" " if w.space_before else "") + w.w.strip() for w in words).strip()


class TranscribeReq(Model):
    audio_key: str
    lang: str = "auto"
    replay_key: str | None = None  # fake provider: replay an existing transcript
    # What the audio should say (back-check of a dubbed clip). Only fakes may use it to invent
    # timed words; a real ASR must NOT see it, or the check would check nothing.
    expected_text: str | None = None
    # Transcribe only these spans, each on its own (one per speaker turn): no word can cross
    # from one speaker into the next.
    clips: list[tuple[int, int]] = Field(default_factory=list)


class BoundaryReq(Model):
    """Ask for sentence-boundary probabilities over a sequence of ASR words."""

    lang: str
    words: list[str]  # word texts in time order
    space_before: list[bool] = Field(default_factory=list)  # for scripts without spaces
    joiner: str = " "
    pauses_ms: list[int] = Field(default_factory=list)  # silence after word i (LLM boundaries)


class Boundaries(Model):
    probs: list[float]  # P(a sentence ends right after word i), one per word


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


class EmbedReq(Model):
    """Speaker embeddings of spans of one audio file (the whole file when `spans` is empty)."""

    audio_key: str
    spans: list[tuple[int, int]] = Field(default_factory=list)  # (start_ms, end_ms)


class Embeddings(Model):
    vectors: list[list[float]]  # one unit vector per span


class VoiceProfileReq(Model):
    """Age and gender from the voice: clean spans of each speaker (one speaker per span)."""

    audio_key: str  # 16 kHz vocals
    spans: list[tuple[int, int]]  # (start_ms, end_ms)
    speakers: list[str]  # speaker of each span


class VoiceProfiles(Model):
    voices: list[SpeakerVoice]  # one per speaker that had spans
    detail: dict[str, dict[str, float]] = Field(default_factory=dict)  # speaker -> raw scores


# ---------------------------------------------------------------- segmentation


class SegmentParams(Model):
    max_unit_ms: int = 15000  # a dub unit is never longer than this
    target_unit_ms: int = 7000  # the "nice" length the splitter aims for
    min_unit_ms: int = 1000  # shorter units get merged when possible
    hard_pause_ms: int = 1500  # a longer silence is a mandatory boundary
    joiner: str = " "  # "" for zh/ja/th
    threshold: float = 0.5  # boundary probability at which a sentence ends


class SegmentReq(Model):
    lang: str
    words: list[Word]
    turns: list[Turn]
    params: SegmentParams
    probs: list[float] = Field(default_factory=list)  # boundary probability per word (or rules)
    speech: list[tuple[int, int]] = Field(default_factory=list)  # VAD regions to snap slots to


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
    parent_ids: list[str] = Field(default_factory=list)  # set when a split/merge made it

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
    feedback: dict[str, str] = Field(default_factory=dict)  # unit id -> problem to fix


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
    self_term: str | None = None  # forms of address the line must keep (from the translation)
    other_term: str | None = None


class RewriteRes(Model):
    text: str


# ---------------------------------------------------------------- TTS


class SynthReq(Model):
    unit_id: str
    text: str
    lang: str
    speaker: str
    voice_key: str | None = None  # 3-10 s reference clip of this speaker (CAS key)
    voice_text: str | None = None  # transcript of the reference clip
    seed: int = 0
    target_ms: int | None = None  # only used when caps.duration_control == "exact"


class SynthRes(Model):
    unit_id: str
    audio_key: str
    sample_rate: int
    dur_ms: int
    provider: str = ""
    exact: bool = False  # True if the engine honoured target_ms


# ---------------------------------------------------------------- voice references


class RefsReq(Model):
    vocals_key: str  # separated speech, 48 kHz
    units: list[Unit]
    min_ms: int = 3000
    max_ms: int = 10000
    unit_vectors: dict[str, list[float]] = Field(default_factory=dict)  # speaker embeddings
    purity: dict[str, float] = Field(default_factory=dict)  # unit id -> share of own speaker


class VoiceRef(Model):
    speaker: str
    audio_key: str  # mono WAV clip cut on word boundaries
    text: str  # what is said in the clip (some TTS engines need it)
    seconds: float
    centroid: list[float] = Field(default_factory=list)  # mean embedding of the speaker's lines
    self_sim: float = 0.0  # how close the speaker's own lines are to that centroid (mean cos)
    sim: float = 0.0  # this clip's similarity to the centroid


class RefBank(Model):
    refs: list[VoiceRef]

    def for_speaker(self, speaker: str) -> VoiceRef | None:
        return next((r for r in self.refs if r.speaker == speaker), None)


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
    gain_db: float = 0.0  # applied in the mix: the line follows the original's level


class LevelReq(Model):
    """Per-line gains so each dubbed line sits at the level of the original line."""

    vocals_key: str  # separated vocals, 48 kHz
    placements: list[Placement]
    max_dev_db: float = 10.0  # original level, at most this far from the speaker's median
    max_gain_db: float = 24.0  # a quiet original recording may need a large cut


class LevelRes(Model):
    gains_db: dict[str, float]  # unit id -> gain
    src_db: dict[str, float | None]  # active level of the original line (None: silent slot)
    dub_db: dict[str, float | None]  # active level of the TTS clip, before the gain


class StretchReq(Model):
    audio_key: str
    rate: float
    sample_rate: int = 48000
    engine: str = "atempo"  # atempo | rubberband


class AudioRes(Model):
    audio_key: str
    dur_ms: int


class AudioReq(Model):
    audio_key: str


class TrimReq(Model):
    """Cut the silence before and after the voice in a TTS clip."""

    audio_key: str
    pad_ms: int = 20  # air kept on each side


class TrimRes(Model):
    audio_key: str
    dur_ms: int
    lead_ms: int  # silence removed at the start
    tail_ms: int  # silence removed at the end


class VadReq(Model):
    audio_key: str  # 16 kHz mono WAV of the separated vocals


class SpeechRegions(Model):
    regions: list[tuple[int, int]]  # (start_ms, end_ms) where someone speaks


class EncodeReq(Model):
    audio_key: str
    codec: str = "aac"
    bitrate: str = "192k"


class TileReq(Model):
    """Mix one fixed 30 s slice of the timeline: the bed plus every clip overlapping it."""

    bed_key: str
    start_ms: int
    end_ms: int
    placements: list[Placement]
    bed_gain_db: float = 0.0


class Tile(Model):
    start_ms: int
    end_ms: int
    audio_key: str  # 48 kHz stereo WAV, not loudness-normalised


class ConcatReq(Model):
    keys: list[str]


class MasterReq(Model):
    """Final loudness normalisation + AAC encode of the whole dub track."""

    audio_key: str
    loudness_lufs: float = -16.0
    bitrate: str = "192k"


class Cue(Model):
    start_ms: int
    end_ms: int
    text: str


class SubItem(Model):
    """One line on the timeline, to be laid out and split into cues."""

    start_ms: int
    end_ms: int
    text: str
    words: list[Word] = Field(default_factory=list)  # timed words of the audio (absolute ms)


class SubsReq(Model):
    lang: str
    items: list[SubItem]
    cpl: int = 42  # width per line (see dub.ops.script.display_len)
    cps: float = 17.0
    max_lines: int = 2
    min_cue_ms: int = 833  # 5/6 s, when there is room
    linger_ms: int = 500  # a cue may stay up this long after the voice stops
    gap_ms: int = 84  # 2 frames at 24 fps between cues


class SubtitleTrack(Model):
    lang: str
    cues: list[Cue]
    srt_key: str


# ---------------------------------------------------------------- human edits

EditField = Literal[
    "text", "seed", "speaker", "nudge", "lock", "split", "merge_next", "undo", "cast"
]


class Edit(Model):
    """One human edit, appended to the job's edits.jsonl. Later edits win.

    text        value: str           the dubbed line, replaces the translation (and is locked)
    seed        value: int           a new take of the same line (regenerate)
    speaker     value: str           who says the line (picks the cloned voice)
    nudge       value: {start_ms, end_ms}  offsets added to the line's time slot
    lock        value: bool          protect the line from automatic rewrites
    split       value: {at_ms} | None  split the unit at the word boundary nearest at_ms
    merge_next  value: None          merge the unit with the next one (same speaker)
    undo        value: edit id       cancel an earlier edit
    cast        value: {speaker, gender?, name?, age_band?, role?}  or
                       {speaker, listener, lang, self_term?, other_term?, since_ms?}
                (unit_id "cast") correct the cast sheet; corrected rows are locked
    """

    id: str  # sortable: <time_ns>-<random>
    unit_id: str
    field: EditField
    value: Any = None
    ts: float = 0.0
    author: str = "human"


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
    BoundaryReq,
    Boundaries,
    DiarizeReq,
    Turn,
    SpeakerVoice,
    Diarization,
    EmbedReq,
    Embeddings,
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
    AudioReq,
    TrimReq,
    TrimRes,
    VadReq,
    SpeechRegions,
    Cue,
    SubItem,
    SubsReq,
    SubtitleTrack,
    Rejection,
    RouteTrace,
    TileReq,
    Tile,
    ConcatReq,
    MasterReq,
    Edit,
    RefsReq,
    VoiceRef,
    RefBank,
    EncodeReq,
]


# Every model-backed op: (request type, result type). Workers and the core share this table.
OPS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "audio.separate": (SeparateReq, Stems),
    "asr.transcribe": (TranscribeReq, WordTimeline),
    "asr.backcheck": (TranscribeReq, WordTimeline),
    "text.boundaries": (BoundaryReq, Boundaries),
    "audio.diarize": (DiarizeReq, Diarization),
    "audio.embed": (EmbedReq, Embeddings),
    "audio.voice_profile": (VoiceProfileReq, VoiceProfiles),
    "mt.context": (ContextReq, ContextSheet),
    "mt.translate": (TranslateReq, TranslationBatch),
    "mt.rewrite": (RewriteReq, RewriteRes),
    "tts.synthesize": (SynthReq, SynthRes),
}
