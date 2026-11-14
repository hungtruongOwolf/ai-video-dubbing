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
    env: str | None = None  # http runtime: name of the worker env (one process per env)
    preload: bool = True  # http runtime: load the model when the worker starts
    requires_bin: list[str] = Field(default_factory=list)  # programs that must be on PATH
    impl_version: str = "1"  # bump when an adapter change alters its output
    weights: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
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
    segment: SegmentCfg = Field(default_factory=SegmentCfg)
    subs: SubsCfg = Field(default_factory=SubsCfg)
    chains: dict[str, list[str]] = Field(default_factory=dict)  # op -> preferred providers
    # Pronouns that are wrong where a relation calls for kinship terms (dub.ops.address).
    address_generic: list[str] = Field(default_factory=list)
    # Two plain sentences in the language: each voice speaks them once so we can measure the
    # silence the TTS adds inside the length we ask for, and ask for that much more.
    calibration: list[str] = Field(default_factory=list)

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


class QaCfg(Cfg):
    cer_max: float = 0.15  # back-check character error rate
    sim_ratio_min: float = 0.5  # voice similarity / the speaker's own similarity
    other_margin: float = 0.05  # fail when another speaker's centroid is closer by this much
    voice_min_ms: int = 1200  # shorter takes are not judged on voice
    tail_min: float = 0.6  # share of the line's last quarter that the ASR heard


class MixCfg(Cfg):
    bed_gain_db: float = 0.0
    loudness_lufs: float = -16.0


class ThroughputCfg(Cfg):
    # Units voiced at the same time. Workers batch what is queued, so this is also the
    # largest batch the TTS can see (with its manifest's limits.max_batch).
    units_in_flight: int = 8
    # After phase A, unload the models only phase A uses (separation, source ASR, sentence
    # boundaries) so the TTS batches have room. "auto": on GPUs under 40 GB.
    free_phase_a_models: bool | Literal["auto"] = "auto"


class TranslateCfg(Cfg):
    window_units: int = 12  # units per LLM call
    context_units: int = 3  # neighbouring units sent along as context


class RouteRule(Cfg):
    id: str
    when: dict[str, Any] = Field(default_factory=dict)
    use: list[str] | None = None
    require: dict[str, Any] = Field(default_factory=dict)


class DubConfig(Cfg):
    version: int = 1
    profile: str = "draft"
    profiles: dict[str, Profile] = Field(default_factory=lambda: {"draft": Profile()})
    timing: TimingCfg = Field(default_factory=TimingCfg)
    qa: QaCfg = Field(default_factory=QaCfg)
    mix: MixCfg = Field(default_factory=MixCfg)
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
