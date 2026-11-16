"""Load the YAML configuration from the `configs/` directory.

configs/
  dub.yaml              global settings: profiles, timing, routing rules, defaults
  langpacks/<lang>.yaml one file per language: speaking rate, subtitles, preferred chains
  providers/<id>.yaml   one manifest per model: which ops it serves and its capabilities
  prompts/*.md          LLM prompts (the prompt text is part of the cache key)

Adding a language = adding a langpack. Adding a model = adding a manifest (+ an adapter).
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dub.contracts.v1 import SegmentParams


class Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- providers


class Caps(Cfg):
    langs: list[str] | Literal["*"] = "*"
    duration_control: Literal["exact", "relative", "rate", "none", "preserving"] = "none"
    max_out_s: float | None = None
    needs_ref_text: bool = False
    sample_rate: int | None = None


class Limits(Cfg):
    max_concurrency: int = 4  # calls in flight from one job (the executor's semaphore)
    max_batch: int = 1  # worker: queued calls run together when the provider has run_batch


class OnError(Cfg):
    usage_limit: Literal["fail", "pause_then_retry"] = "fail"
    retry_wait_s: list[float] = Field(default_factory=lambda: [30.0, 120.0])


class ProviderManifest(Cfg):
    id: str
    op: list[str]
    runtime: Literal["inproc", "cli", "http"] = "inproc"
    impl: str | None = None  # "module.path:ClassName" for the inproc runtime
    env: str | None = None  # http runtime: name of the worker env (its venv, pod/envs/<env>.txt)
    # The worker process (default: the env's). A provider given its own worker runs in its own
    # process from the same venv, so its calls do not queue behind the env's other models (M10:
    # a 10 ms speaker embedding waited behind OmniVoice batches, back-checks behind Whisper).
    worker: str | None = None
    preload: bool = True  # http runtime: load the model when the worker starts
    requires_bin: list[str] = Field(default_factory=list)  # programs that must be on PATH
    impl_version: str = "1"  # bump when an adapter change alters its output
    # packages pinned in pod/envs/<env>.txt whose versions are part of every result key
    libs: list[str] = Field(default_factory=list)
    weights: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    # per-op params on top of `params` (e.g. a stronger LLM for one op only)
    op_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    prompt_files: dict[str, str] = Field(default_factory=dict)  # op -> file in prompts/
    caps: Caps = Field(default_factory=Caps)
    limits: Limits = Field(default_factory=Limits)
    on_error: OnError = Field(default_factory=OnError)
    status: Literal["active", "candidate"] = "active"
    fake: bool = False  # fake provider, selected by --fake
    license: str | None = None
    description: str = ""

    @property
    def weights_rev(self) -> str | None:
        if not self.weights:
            return None
        return str(self.weights.get("revision") or self.weights.get("hf") or "")


# ---------------------------------------------------------------- langpacks


def worker_of(m: ProviderManifest) -> str:
    """The worker process that hosts an http provider: its own, or its env's."""
    return m.worker or m.env or ""


class SegmentCfg(Cfg):
    max_unit_s: float = 15.0
    target_unit_s: float = 7.0
    min_unit_s: float = 1.0
    hard_pause_s: float = 1.5
    threshold: float = 0.5  # calibrated boundary probability at which a sentence ends


class SubsCfg(Cfg):
    cpl: int = 42  # characters per line
    cps: float = 17.0  # characters per second
    max_lines: int = 2


class Langpack(Cfg):
    lang: str
    name: str = ""
    iso639_2: str = "und"  # 3-letter code for container metadata
    speech_unit: Literal["syllable", "char", "word"] = "syllable"
    units_per_s: float = 4.5  # average speaking rate (estimate; E7 in M3 measures it)
    joiner: str = " "  # "" for zh/ja/th
    # M8: a short line takes longer than its units at the average rate (the onset, the
    # lengthened last syllable). Its planned length is multiplied by short_boost up to
    # short_units units, the boost fading to none at twice that. 1.0 = linear only.
    short_boost: float = 1.0
    short_units: int = 0
    segment: SegmentCfg = Field(default_factory=SegmentCfg)
    subs: SubsCfg = Field(default_factory=SubsCfg)
    chains: dict[str, list[str]] = Field(default_factory=dict)  # op -> preferred providers
    # Pronouns that are wrong where a relation calls for kinship terms (dub.ops.address).
    address_generic: list[str] = Field(default_factory=list)
    # Two plain sentences in the language: each voice speaks them once so we can measure the
    # silence the TTS adds inside the length we ask for, and ask for that much more.
    calibration: list[str] = Field(default_factory=list)
    # A line made only of these (laughter, a cry, a filler: 哈哈, 啊, "uh-huh") keeps its
    # original audio: dubbed, "Ha" sounds fake and rarely fits a 20 ms slot. A regex matched
    # against the whole line, lower-cased, without spaces and punctuation; "" = none.
    interjections: str = ""
    # Sentence-final particles the back-check ASR swaps for one another with no change of
    # meaning or of mouth shape ("Nó ăn được à?" heard as "nó ăn được hả"): a line ending in
    # one that is heard ending in another of its group is not misheard (dub.ops.qa).
    final_particles: list[list[str]] = Field(default_factory=list)

    def is_interjection(self, text: str) -> bool:
        if not self.interjections:
            return False
        bare = re.sub(r"[\W_]+", "", unicodedata.normalize("NFC", text).lower())
        return bool(bare) and re.fullmatch(self.interjections, bare) is not None

    def segment_params(self) -> SegmentParams:
        s = self.segment
        return SegmentParams(
            max_unit_ms=int(s.max_unit_s * 1000),
            target_unit_ms=int(s.target_unit_s * 1000),
            min_unit_ms=int(s.min_unit_s * 1000),
            hard_pause_ms=int(s.hard_pause_s * 1000),
            joiner=self.joiner,
            threshold=s.threshold,
        )


# ---------------------------------------------------------------- dub.yaml


class Profile(Cfg):
    tts_chain: Literal["incumbent_only", "langpack"] = "langpack"
    llm_rewrite: bool = False
    max_rewrites: int = 2
    backcheck: bool = True  # transcribe every dubbed line again: subtitle timing + CER
    takes: int = 1  # QA gate: up to this many takes (seeds) per line; 1 = check and flag only
    # Per-provider params laid over the manifest's, e.g. {omnivoice: {num_step: 16}}.
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TimingCfg(Cfg):
    variant_window: float = 0.08  # a variant predicted within ±8% of the slot "fits"
    safe_ratio: tuple[float, float] = (0.85, 1.10)  # how far an exact engine may be squeezed (E3)
    rate_p95: float = 1.10  # faster than this => try a rewrite (if the profile allows)
    rate_cap: float = 1.20  # hard ceiling; beyond it the unit is flagged "overflow"
    early_start_ms: int = 200  # max early start
    late_end_ms: int = 500  # max late end
    min_pause_ms: int = 120  # minimum pause kept between two lines
    tail_guard_ms: int = 150  # space kept before the next line
    strict: bool = False  # tightly synced content: halve the borrowing limits
    # M11: a take pauses where its actor pauses (dub.ops.phrases): pieces of the take cut at
    # its silences, each sped up or slowed down within phrase_rate
    phrase_align: bool = True
    phrase_pause_ms: int = 250  # the actor's pauses shorter than this are not pauses
    phrase_rate: tuple[float, float] = (0.87, 1.15)


class QaCfg(Cfg):
    cer_max: float = 0.15  # back-check character error rate
    sim_ratio_min: float = 0.5  # voice similarity / the speaker's own similarity
    other_margin: float = 0.05  # fail when another speaker's centroid is closer by this much
    twin_sim: float = 0.8  # ...unless the two centroids are this close (similar voices)
    # M8: fail a take that Whisper (language auto) hears as the SOURCE language: a cloned
    # voice carrying its actor's accent. Takes shorter than lid_min_ms are not judged
    # (language identification is unreliable on a second of speech).
    source_lid: bool = False
    lid_min_ms: int = 1500
    voice_min_ms: int = 1200  # shorter takes are not judged on voice
    tail_min: float = 0.6  # share of the line's last quarter that the ASR heard


class MixCfg(Cfg):
    bed_gain_db: float = 0.0
    loudness_lufs: float = -16.0
    # each line at the level of the original line (separated vocals), see ops/level.py
    match_level: bool = True
    level_max_dev_db: float = 10.0
    level_max_gain_db: float = 24.0
    # Laughter, cries, screams between the lines (no words heard in them) stay in the dub
    # (dub.ops.nonverbal); gaps shorter than nonverbal_min_ms are left out.
    keep_nonverbal: bool = True
    # Scale the separated stems so that they add up to the mix (the separator normalises each)
    rebalance_stems: bool = True
    nonverbal_min_ms: int = 200  # M10: a 250 ms cry between the song and a line was left out
    nonverbal_pad_ms: int = 150


class ThroughputCfg(Cfg):
    # Units voiced at the same time. Workers batch what is queued, so this is also the
    # largest batch the TTS can see (with its manifest's limits.max_batch).
    units_in_flight: int = 8
    # After phase A, unload the models only phase A uses (separation, source ASR, sentence
    # boundaries) so the TTS batches have room. "auto": on GPUs under 30 GB (24 GB cards).
    free_phase_a_models: bool | Literal["auto"] = "auto"
    # When take 0 of a line fails QA, voice its other takes (and other wordings) at once
    # instead of one after another: shorter retake chains, more work queued at once.
    parallel_retakes: bool = False  # M10 A/B: slower (more work queued, same choices)
    # dubd: jobs analysing at once (separation, ASR, speakers); the others translate or voice
    analyses_at_once: int = 1


class TranslateCfg(Cfg):
    window_units: int = 12  # units per LLM call
    context_units: int = 3  # neighbouring units sent along as context


class RouteRule(Cfg):
    id: str
    when: dict[str, Any] = Field(default_factory=dict)
    use: list[str] | None = None
    require: dict[str, Any] = Field(default_factory=dict)


class VoiceCfg(Cfg):
    # Reference clips auditioned per speaker before voicing (1 = take the most typical clip).
    # Each says the langpack's two calibration sentences; the clip whose takes sound most like
    # the speaker (embedding similarity), among the intelligible ones, is kept.
    audition: int = 1  # off by default: no gain measured (M7, docs/m7-quality-pass.md)
    audition_cer_slack: float = 0.05  # intelligible: back-check CER within this of the best
    # M12: each line voiced from the speaker's clean clip whose delivery (pitch, range, level,
    # voicing) is closest to the line's own, out of up to `styles` clips (ops/prosody.py); a
    # pool clip only when closer than the usual reference by `style_margin` (z units)
    emotion_refs: bool = True
    styles: int = 12
    style_margin: float = 0.5
    # M13: what "closest delivery" means: prosody (pitch, range, level, voicing), emotion
    # (emotion2vec embeddings) or both; in the last two a clip must beat the usual reference
    # by style_margin_rel (median units)
    style_by: str = "prosody"
    style_margin_rel: float = 0.25


class SpeakersCfg(Cfg):
    # Readings of the script by the LLM (text.speakers), voted line by line (ops/speakers.py).
    votes: int = 3


class DubConfig(Cfg):
    version: int = 1
    profile: str = "draft"
    profiles: dict[str, Profile] = Field(default_factory=lambda: {"draft": Profile()})
    timing: TimingCfg = Field(default_factory=TimingCfg)
    qa: QaCfg = Field(default_factory=QaCfg)
    mix: MixCfg = Field(default_factory=MixCfg)
    voice: VoiceCfg = Field(default_factory=VoiceCfg)
    speakers: SpeakersCfg = Field(default_factory=SpeakersCfg)
    translate: TranslateCfg = Field(default_factory=TranslateCfg)
    throughput: ThroughputCfg = Field(default_factory=ThroughputCfg)
    defaults: dict[str, list[str]] = Field(default_factory=dict)
    routes: dict[str, list[RouteRule]] = Field(default_factory=dict)


# ---------------------------------------------------------------- loader

# Ops whose chain comes from the SOURCE language; every other op uses the TARGET language.
SOURCE_SIDE_OPS = {
    "audio.separate",
    "asr.transcribe",
    "asr.align",
    "audio.diarize",
    "text.boundaries",
    "text.speakers",
}
# An op without its own chain in a langpack or in `defaults:` uses the chain of this op.
CHAIN_FALLBACK = {"asr.backcheck": "asr.transcribe"}


class Config:
    def __init__(
        self,
        root: Path,
        dub: DubConfig,
        langpacks: dict[str, Langpack],
        providers: dict[str, ProviderManifest],
    ):
        self.root = root
        self.dub = dub
        self.langpacks = langpacks
        self.providers = providers

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    def langpack(self, lang: str) -> Langpack:
        if lang in self.langpacks:
            return self.langpacks[lang]
        base = self.langpacks.get("default", Langpack(lang="default"))
        return base.model_copy(update={"lang": lang})

    def profile(self, name: str | None = None) -> Profile:
        name = name or self.dub.profile
        if name not in self.dub.profiles:
            raise KeyError(f"Profile '{name}' is not defined in dub.yaml")
        return self.dub.profiles[name]

    def fake_provider_for(self, op: str) -> str | None:
        for p in self.providers.values():
            if p.fake and op in p.op:
                return p.id
        return None


def default_config_dir() -> Path:
    env = os.environ.get("DUB_CONFIG_DIR")
    if env:
        return Path(env)
    # src/dub/core/config.py -> <repo root>/configs
    return Path(__file__).resolve().parents[3] / "configs"


def _load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(root: Path | None = None) -> Config:
    root = Path(root) if root else default_config_dir()
    if not (root / "dub.yaml").exists():
        raise FileNotFoundError(f"{root / 'dub.yaml'} not found (set DUB_CONFIG_DIR?)")
    dub = DubConfig.model_validate(_load_yaml(root / "dub.yaml"))
    langpacks: dict[str, Langpack] = {}
    for p in sorted((root / "langpacks").glob("*.yaml")):
        lp = Langpack.model_validate(_load_yaml(p))
        langpacks[lp.lang] = lp
    providers: dict[str, ProviderManifest] = {}
    for p in sorted((root / "providers").glob("*.yaml")):
        m = ProviderManifest.model_validate(_load_yaml(p))
        if m.id in providers:
            raise ValueError(f"Duplicate provider id '{m.id}' ({p})")
        providers[m.id] = m
    return Config(root=root, dub=dub, langpacks=langpacks, providers=providers)
