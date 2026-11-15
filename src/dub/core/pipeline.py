"""The standard cascade: audio -> separate -> ASR -> diarize -> segment -> context (cast
sheet) -> translate -> TTS -> timing fit -> mix + subtitles.

This runs where the models are: on the laptop with fake providers in M0, on the GPU pod
from M1. It only ever sees the AUDIO track. The laptop keeps the video and does the final
mux (see `dub.laptop`).

Phases (from the design):
  Phase A  analysis of the whole file (separate, ASR, diarize, segment)
  Barrier  one LLM context pass over the whole transcript: summary, glossary, cast sheet
  Phase B  per-unit streaming: translate windows -> TTS -> fit -> stretch; then mix
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dub.contracts.v1 import (
    Attribution,
    AudioReq,
    AudioRes,
    Boundaries,
    BoundaryReq,
    ConcatReq,
    ContextReq,
    ContextSheet,
    CutReq,
    DecodeReq,
    Diarization,
    DiarizeReq,
    Edit,
    Embeddings,
    EmbedReq,
    HardsubCue,
    HardsubFix,
    HardsubReq,
    LineLevel,
    LineLevelReq,
    MasterReq,
    MediaInfo,
    Placement,
    RebalanceReq,
    RefBank,
    RefsReq,
    RewriteReq,
    Segmentation,
    SegmentReq,
    SeparateReq,
    SpeakersReq,
    SpeakerVoice,
    SpeechRegions,
    Stems,
    StretchReq,
    SubItem,
    SubsReq,
    SubtitleTrack,
    SynthReq,
    SynthRes,
    Tile,
    TileReq,
    TranscribeReq,
    TranslateReq,
    Translation,
    TranslationBatch,
    TrimReq,
    TrimRes,
    Turn,
    Unit,
    VadReq,
    VoiceProfileReq,
    VoiceProfiles,
    VoiceRef,
    Word,
    WordTimeline,
    join_words,
)
from dub.core.config import Config, Langpack, ProviderManifest
from dub.core.ctx import Ctx
from dub.core.executor import Executor
from dub.core.provider_api import ProviderError
from dub.core.router import Router, RoutingError
from dub.core.trace import Tracer
from dub.ops import address as address_ops
from dub.ops import diarize as diar_ops
from dub.ops import edits as edits_ops
from dub.ops import hardsubs as hardsub_ops
from dub.ops import level as level_ops
from dub.ops import media as media_ops
from dub.ops import mix as mix_ops
from dub.ops import nonverbal as nonverbal_ops
from dub.ops import qa as qa_ops
from dub.ops import refs as refs_ops
from dub.ops import risk as risk_ops
from dub.ops import segment as segment_ops
from dub.ops import speakers as speaker_ops
from dub.ops import subs as subs_ops
from dub.ops import vad as vad_ops
from dub.ops.align import cer
from dub.ops.script import base_lang
from dub.ops.timing import (
    Fit,
    Neighbor,
    count_units,
    exact_target_ms,
    fit_unit,
    pick_variant,
    predict_ms,
)
from dub.store import Workspace


@dataclass
class JobSpec:
    audio_key: str  # CAS key of the original audio track (sent by the laptop)
    source_name: str  # file name, used for the job id and the report
    tgt_lang: str
    src_lang: str = "auto"
    profile: str | None = None
    transcript_key: str | None = None  # optional transcript to replay (fake ASR, tests)
    speakers_hint: int | None = None
    subs_only: bool = False  # path P5: subtitles only, no TTS
    edits: list[Edit] = field(default_factory=list)  # human edits (see dub.ops.edits)
    export: bool = True  # False: stop after the mixed tiles (fast preview while editing)
    analysis_only: bool = False  # stop after segmentation (benchmarks: dub bench seg)
    confirm_cast: bool = False  # stop after the cast sheet, for the user to correct it
    fresh: bool = False  # recompute every step, ignoring the cache (dub bench speed --fresh)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)  # provider -> overrides
    hardsubs: list[HardsubCue] = field(default_factory=list)  # burned-in subtitles (laptop OCR)


@dataclass
class JobResult:
    job_id: str
    job_dir: Path
    src_lang: str
    tgt_lang: str
    dub_audio_key: str | None
    dub_m4a_key: str | None  # same audio as AAC, small enough to download
    subs: dict[str, str]  # lang -> CAS key of the .srt
    placements: list[Placement] = field(default_factory=list)
    tiles: list[Tile] = field(default_factory=list)
    a48k_key: str | None = None  # original audio, 48 kHz (for side-by-side previews)
    warnings: list[str] = field(default_factory=list)
    redone: dict[str, int] = field(default_factory=dict)  # op -> calls that actually ran
    metrics: dict[str, Any] = field(default_factory=dict)  # sync and quality numbers (report)
    calls: int = 0
    cache_hits: int = 0


def parse_params(items: list[str]) -> dict[str, dict[str, Any]]:
    """["whisper_large_v3.batch_size=16"] -> {"whisper_large_v3": {"batch_size": 16}}.
    Values are read as JSON when they parse (numbers, true/false), else kept as text."""
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        target, _, value = item.partition("=")
        pid, _, key = target.partition(".")
        if not pid or not key or not _:
            raise ValueError(f"--param wants PROVIDER.KEY=VALUE, got {item!r}")
        try:
            v: Any = json.loads(value)
        except json.JSONDecodeError:
            v = value
        out.setdefault(pid, {})[key] = v
    return out


def with_settings(cfg: Config, settings: dict[str, Any]) -> Config:
    """dub.yaml settings overridden for one run: --param dub.voice.audition=1 gives
    {"voice.audition": 1}. Unknown keys are an error, like in dub.yaml."""
    data = cfg.dub.model_dump()
    for path, value in settings.items():
        *parents, leaf = path.split(".")
        node = data
        for k in parents:
            if not isinstance(node.get(k), dict):
                raise ValueError(f"--param dub.{path}: no section {k!r} in dub.yaml")
            node = node[k]
        if leaf not in node:
            raise ValueError(f"--param dub.{path}: no setting {leaf!r} in dub.yaml")
        node[leaf] = value
    return Config(cfg.root, type(cfg.dub).model_validate(data), cfg.langpacks, cfg.providers)


def job_id_for(spec: JobSpec, profile: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(spec.source_name).stem).strip("-").lower() or "job"
    return f"{stem[:40]}-{spec.tgt_lang}-{profile}-{spec.audio_key[:8]}"


async def run_job(
    cfg: Config,
    ws: Workspace,
    spec: JobSpec,
    *,
    overrides: list[tuple[str, list[str]]] | None = None,
    log: Callable[[str], None] = print,
    phase_a_done: Callable[[], None] | None = None,
) -> JobResult:
    """One job. `phase_a_done` is called when the analysis (separation, ASR, speakers,
    sentences) is over: dubd lets the next job's analysis start then (several videos at once)."""
    # Every worker call and CPU step holds a thread while it waits (asyncio.to_thread). The
    # default pool (32 threads) was full 29-40% of the voicing time, so fewer lines reached
    # the GPU workers to batch than were in flight.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=JOB_THREADS, thread_name_prefix="job")
    )
    if "dub" in spec.params:  # --param dub.SECTION.KEY=VALUE: a dub.yaml setting, this run only
        cfg = with_settings(cfg, spec.params["dub"])
        spec = dataclasses.replace(
            spec, params={k: v for k, v in spec.params.items() if k != "dub"}
        )
    profile_name = spec.profile or cfg.dub.profile
    profile = cfg.profile(profile_name)
    t = cfg.dub.timing
    job_id = job_id_for(spec, "analysis" if spec.analysis_only else profile_name)
    job_dir = ws.job_dir(job_id)
    tracer = Tracer()
    router = Router(cfg, profile=profile_name, overrides=overrides)
    cx = Ctx(
        cfg=cfg,
        ws=ws,
        router=router,
        executor=Executor(cfg, ws.cas, log=log),
        tracer=tracer,
        job_id=job_id,
        log=log,
        facts={"src_lang": spec.src_lang, "tgt_lang": spec.tgt_lang, "profile": profile_name},
        fresh=spec.fresh,
        params=spec.params,
    )
    ws.db.upsert_job(
        job_id,
        source=spec.source_name,
        src_lang=spec.src_lang,
        tgt_lang=spec.tgt_lang,
        profile=profile_name,
        status="running",
    )

    def stage(msg: str) -> None:
        """A pipeline stage starts: log it, and mark it on the trace (dub bench speed)."""
        log(msg)
        tracer.instant(msg, "stages")

    # ---------------------------------------------------------------- Phase A
    stage("[1/7] decoding audio")
    media: MediaInfo = await cx.local(
        "media.decode", media_ops.decode, DecodeReq(src_key=spec.audio_key), MediaInfo
    )
    stage("[2/7] separating speech from background + transcribing (in parallel)")
    stems, words = await asyncio.gather(
        cx.call("audio.separate", SeparateReq(audio_key=media.a48k_key)),
        cx.call(
            "asr.transcribe",
            TranscribeReq(
                audio_key=media.a16k_key, lang=spec.src_lang, replay_key=spec.transcript_key
            ),
        ),
    )
    stems: Stems
    words: WordTimeline
    if cfg.dub.mix.rebalance_stems:
        stems = await cx.local(
            "audio.rebalance",
            mix_ops.rebalance,
            RebalanceReq(mix_key=media.a48k_key, vocals_key=stems.vocals_key,
                         instrumental_key=stems.instrumental_key),
            Stems,
        )  # fmt: skip
    src_lang = words.lang
    cx.facts["src_lang"] = src_lang
    src_lp, tgt_lp = cfg.langpack(src_lang), cfg.langpack(spec.tgt_lang)

    stage("[3/7] finding who speaks when")
    vocals16: AudioRes = await cx.local(
        "audio.to_mono16k", media_ops.to_mono16k, AudioReq(audio_key=stems.vocals_key), AudioRes
    )
    diar: Diarization = await cx.call(
        "audio.diarize",
        DiarizeReq(
            audio_key=vocals16.audio_key,
            speakers_hint=spec.speakers_hint,
            replay_key=spec.transcript_key,
        ),
    )
    if len(diar.speakers) > 1 and not spec.transcript_key:
        # Several speakers: transcribe again turn by turn. A whole-file pass merges words
        # across turns ("...keep my promises dad" took the next speaker's "Dad,") and the
        # translator then picks the wrong listener (measured on the drama, E10).
        log(f"      {len(diar.speakers)} speakers: transcribing turn by turn")
        words = await cx.call(
            "asr.transcribe",
            TranscribeReq(
                audio_key=media.a16k_key,
                lang=src_lang,
                clips=segment_ops.turn_clips(diar.turns, media.duration_ms),
            ),
        )
    asr_words = segment_ops.normalize_words(
        sorted((w for w in words.words if w.w.strip()), key=lambda w: (w.start_ms, w.end_ms))
    )
    hardsub_stats: dict[str, Any] = {}
    if spec.hardsubs and base_lang(src_lang) == "zh":  # the subtitles in the picture win
        fix: HardsubFix = await cx.local(
            "text.hardsubs", hardsub_ops.correct,
            HardsubReq(lang=src_lang, words=asr_words, cues=spec.hardsubs), HardsubFix,
            with_cas=False,
        )  # fmt: skip
        asr_words = fix.words
        hardsub_stats = {"hardsub_cues": len(spec.hardsubs), "hardsub_corrected": fix.corrected,
                         "hardsub_inserted": fix.inserted,
                         "hardsub_coverage": fix.coverage}  # fmt: skip
        missed = f", {fix.inserted} lines the ASR missed added" if fix.inserted else ""
        log(f"      burned-in subtitles: {fix.corrected} characters of the transcript corrected"
            f"{missed} (the subtitles cover {fix.coverage:.0%} of the speech)")  # fmt: skip
    # M8: who says each line, read from the script. The diarizer merges similar voices and
    # misses quick exchanges; the turns it gave are replaced by one per line of dialogue.
    speaker_stats: dict[str, Any] = {}
    if not spec.speakers_hint and not spec.transcript_key:
        covered = hardsub_stats.get("hardsub_coverage", 0) >= hardsub_ops.CAPTIONED_MIN
        cues = spec.hardsubs if covered else None
        diar, speaker_stats = await _attribute_speakers(
            cx, diar, asr_words, cues, src_lang, src_lp.joiner, vocals16.audio_key, log
        )
    speech: SpeechRegions = await cx.local(
        "audio.vad",
        vad_ops.speech_regions,
        VadReq(audio_key=vocals16.audio_key),
        SpeechRegions,
    )
    bounds: Boundaries = await cx.call(
        "text.boundaries",
        BoundaryReq(
            lang=src_lang,
            words=[w.w for w in asr_words],
            space_before=[w.space_before for w in asr_words] if not src_lp.joiner else [],
            joiner=src_lp.joiner,
            pauses_ms=_pauses(asr_words) if _wants_pauses(cx, "text.boundaries") else [],
        ),
    )
    seg: Segmentation = await cx.local(
        "text.segment",
        segment_ops.segment,
        SegmentReq(
            lang=src_lang,
            words=asr_words,
            turns=diar.turns,
            params=src_lp.segment_params(),
            probs=bounds.probs,
            speech=speech.regions,
            smooth=not speaker_stats.get("applied"),
        ),
        Segmentation,
        with_cas=False,
    )
    base_units = seg.units
    if spec.analysis_only:
        if phase_a_done:
            phase_a_done()
        n_words, n_sent = len(words.words), len(seg.sentences)
        log(f"      {n_words} words -> {n_sent} sentences -> {len(seg.units)} units")
        return _analysis_result(cx, tracer, ws, job_id, job_dir, spec, seg, src_lang, media)
    edits = edits_ops.effective(spec.edits)
    ue = edits_ops.UnitEdits.from_edits(edits)
    worded = segment_ops.assign_speakers(asr_words, diar.turns)
    units, warnings = edits_ops.apply_structure(base_units, worded, edits, src_lp.joiner)
    for w in warnings:
        log(f"      edit skipped: {w}")
    if speaker_stats.get("failed"):
        warnings = [*warnings, "speaker attribution failed: speakers from the voices alone"]
    voiced = ue.apply_to_units(units)  # speaker and slot edits
    auto = _not_dialogue(voiced, spec.hardsubs, hardsub_stats, speaker_stats)
    cries = [u.id for u in voiced if src_lp.is_interjection(u.text) and u.id not in auto]
    cries_set = {
        u.id for u in voiced if src_lp.is_interjection(u.text) and u.id not in ue.dub_forced
    }
    auto = [uid for uid in auto + cries if uid not in ue.dub_forced]
    ue.original.update(auto)
    if cries:
        speaker_stats["interjections_original"] = len(cries)
    if hardsub_stats or speaker_stats.get("applied") or cries:
        speaker_stats["auto_original"] = len(auto)
        if auto:
            log(f"      {len(auto)} lines kept original (songs, background voices, "
                f"{len(cries)} laughs and cries)")  # fmt: skip
    # Speaker embedding of every line: picks the most typical reference clip per speaker and
    # gives the QA gate each speaker's centroid. Purity: share of a line spoken by its speaker.
    emb: Embeddings = await cx.call(
        "audio.embed",
        EmbedReq(audio_key=vocals16.audio_key, spans=[(u.start_ms, u.end_ms) for u in voiced]),
    )
    unit_vectors = dict(zip((u.id for u in voiced), emb.vectors, strict=True))
    purity = {
        u.id: 1.0
        if u.id in ue.speaker
        else round(diar_ops.purity(u.start_ms, u.end_ms, u.speaker, diar.turns), 3)
        for u in voiced
    }
    refs: RefBank = await cx.local(
        "voice.extract_refs",
        refs_ops.extract_refs,
        RefsReq(
            vocals_key=stems.vocals_key,
            # lines kept original (a song) never give a speaker their voice
            units=[u for u in voiced if u.id not in ue.original] or voiced,
            unit_vectors=unit_vectors,
            purity=purity,
            # M8: a few clips per speaker; the one whose words the ASR hears as the line wins
            candidates=1 if spec.subs_only else max(REF_CANDIDATES, cfg.dub.voice.audition),
        ),  # fmt: skip
        RefBank,
    )
    if not spec.subs_only:
        refs = await _retranscribe_refs(
            cx, refs, src_lang, src_lp.joiner, log, keep_alternates=cfg.dub.voice.audition > 1
        )
    if edits:
        log(f"      {len(edits)} human edits applied")
    # Voice hints for the cast sheet (age, gender): weak evidence, the dialogue outranks it.
    spans, span_speakers = _voice_spans(voiced, purity)
    if spans:
        try:
            prof: VoiceProfiles = await cx.call(
                "audio.voice_profile",
                VoiceProfileReq(audio_key=vocals16.audio_key, spans=spans, speakers=span_speakers),
            )
            diar = diar.model_copy(update={"voices": prof.voices})
        except (RoutingError, ProviderError) as e:
            log(f"      voice hints skipped: {str(e)[:200]}")
    if _free_phase_a(cfg):  # 24 GB GPUs: room for the TTS batches of phase B
        # Whisper stays when the accent check (language ID of every take) needs it: freed, it
        # was restored again by the first take
        keep = ["asr.backcheck", "tts.synthesize", "audio.embed"]
        keep += ["asr.transcribe"] if cfg.dub.qa.source_lid else []
        freed = await cx.free_models(
            ["audio.separate", "asr.transcribe", "text.boundaries", "audio.voice_profile"],
            keep_ops=keep,
        )
        if freed:
            log(f"      GPU memory freed: {', '.join(freed)}")
    log(f"      {len(words.words)} words -> {len(seg.sentences)} sentences -> {len(units)} units")

    if phase_a_done:
        phase_a_done()

    # ---------------------------------------------------------------- Barrier: context pass
    stage("[4/7] reading the whole script: summary, glossary, cast sheet")
    # Meanwhile, each voice is calibrated: how much silence its takes carry (see below).
    calib = (
        asyncio.create_task(
            _audition_voices(cx, refs, tgt_lp, spec.tgt_lang, cfg.dub.voice.audition_cer_slack)
        )
        if not spec.subs_only and not spec.confirm_cast
        else None
    )
    sheet: ContextSheet = await cx.call(
        "mt.context",
        ContextReq(src_lang=src_lang, tgt_lang=spec.tgt_lang, units=base_units, voices=diar.voices),
    )
    sheet = edits_ops.apply_cast(sheet, edits)  # the user's corrections win
    finer = None if speaker_stats.get("applied") else _finer_diarization(cx, sheet, spec)
    if finer is not None:  # one voice holds several people: cluster again, finer (once)
        if calib is not None:
            calib.cancel()
        merged = ", ".join(f"{s.id} ({s.characters})" for s in sheet.cast.speakers
                           if s.characters > 1)  # fmt: skip
        log(f"      the cast sheet finds several people under one voice: {merged}; "
            f"finding who speaks when again (threshold {finer})")  # fmt: skip
        diar_params = {**spec.params.get("ecapa_cluster", {}), "threshold": finer,
                       "min_share": DIAR_FINER_MIN_SHARE}  # fmt: skip
        spec = dataclasses.replace(spec, params={**spec.params, "ecapa_cluster": diar_params})
        return await run_job(cfg, ws, spec, overrides=overrides, log=log)
    if spec.confirm_cast:
        return _cast_result(cx, tracer, ws, job_id, job_dir, spec, sheet, voiced, src_lang, media,
                            diar.voices, profile_name)  # fmt: skip

    auditions: list[Audition] = []
    tts_pad: dict[str, int] = {}
    if calib is not None:
        refs, tts_pad, auditions = await calib
    for a in auditions:
        sim = a.sims[a.chosen]
        log(
            f"      voice {a.speaker}: clip {a.chosen + 1} of {len(a.sims)}"
            + (
                f" (similarity {sim:.3f}, first clip {a.sims[0]:.3f})"
                if sim is not None and a.sims[0] is not None
                else ""
            )
            + (f", silence pad {a.pad_ms} ms" if a.pad_ms else "")
        )

    # ---------------------------------------------------------------- Phase B: translate
    stage("[5/7] translating")
    tcfg = cfg.dub.translate
    translations: dict[str, Translation] = {}

    # Windows are cut on the base (unedited) units, so a split or merge only changes the
    # window that holds it; neighbours come from the base units too.
    size, ctx_n = tcfg.window_units, tcfg.context_units
    base_index = {u.id: i for i, u in enumerate(base_units)}
    groups: dict[int, list[Unit]] = {}
    for u in units:
        groups.setdefault(base_index.get((u.parent_ids or [u.id])[0], 0) // size, []).append(u)

    q = cfg.dub.qa
    limits = qa_ops.QaLimits(
        cer_max=q.cer_max,
        sim_ratio_min=q.sim_ratio_min,
        tail_min=q.tail_min,
        other_margin=q.other_margin,
        twin_sim=q.twin_sim,
        voice_min_ms=q.voice_min_ms,
    )

    accent_note: list[str] = []  # logged once: why the accent retry could not run

    async def engine_retry(
        u: Unit, text: str, pred: int, seed: int, facts: dict, best, fact: str = "take.accent_retry"
    ):
        """One more take from the engine a retry route names, when the line's own engine
        failed it. `take.accent_retry`: every take carried the actor's accent (heard as the
        source language); M8, experiments/accent.py on a zh drama: OmniVoice clones were
        heard as Chinese on 22% of the lines (18% best of 3 seeds), VieNeu-TTS on 3.7%, at a
        similar likeness (0.51 vs 0.53). `take.qa_retry`: every take failed the QA gate (VieNeu
        read "hứng" as "hơn" in 9 takes out of 9). Kept when it passes the gate or scores
        better; skipped when the route names the engine that already failed."""
        try:
            pid = cx.router.route("tts.synthesize", {**cx.facts, **facts, fact: True}).chain[0]
        except RoutingError:
            return best
        if pid == best[1].provider:
            return best
        try:
            take: SynthRes = await cx.call(
                "tts.synthesize",
                lambda m: synth_request(u, text, pred, seed, m),
                facts={**facts, fact: True},
                subject=u.id,
            )
        except (RoutingError, ProviderError) as e:
            if not accent_note:
                accent_note.append(str(e))
                log(f"      {fact.split('.')[1].replace('_', ' ')} unavailable: {str(e)[:160]}")
            return best
        trim: TrimRes = await cx.local(
            "audio.trim", media_ops.trim, TrimReq(audio_key=take.audio_key), TrimRes,
            subject=u.id,
        )  # fmt: skip
        res, words_r = await qa_take(u, trim.audio_key, text)
        if res.passed or res.score > best[0].score:
            return (res, take, trim, words_r)
        return best

    def lid_ok() -> bool:
        """The source ASR detects the language (PhoWhisper, Kotoba force theirs)."""
        try:
            pid = cx.router.route("asr.transcribe", dict(cx.facts)).chain[0]
        except Exception:  # noqa: BLE001 - no ASR: no accent check
            return False
        return not cx.params_for(cfg.providers[pid], "asr.transcribe").get("language")

    async def qa_take(u: Unit, audio_key: str, text: str) -> tuple[qa_ops.QaResult, list[Word]]:
        """Back-check ASR + speaker embedding of one take, judged against the QA limits."""
        ref = refs.for_speaker(u.speaker)
        clip16: AudioRes = await cx.local(
            "audio.to_mono16k",
            media_ops.to_mono16k,
            AudioReq(audio_key=audio_key),
            AudioRes,
            subject=u.id,
        )

        # The three checks are independent: run them at the same time (awaited one by one they
        # took p50 7.1 s per take on the drama for under 0.5 s of model work).
        async def backcheck() -> WordTimeline:
            return await cx.call(
                "asr.backcheck",
                TranscribeReq(audio_key=clip16.audio_key, lang=spec.tgt_lang, expected_text=text),
                subject=u.id,
            )

        async def embed() -> list[float] | None:
            if not (ref and ref.centroid):
                return None
            e: Embeddings = await cx.call(
                "audio.embed", EmbedReq(audio_key=clip16.audio_key), subject=u.id
            )
            return e.vectors[0]

        async def language() -> WordTimeline | None:
            if not (
                q.source_lid
                and clip16.dur_ms >= q.lid_min_ms
                and base_lang(src_lang) != base_lang(spec.tgt_lang)
                and lid_ok()
            ):
                return None
            try:
                return await cx.call(
                    "asr.transcribe",
                    TranscribeReq(audio_key=clip16.audio_key, lang="auto", detect_only=True),
                    subject=u.id,
                )
            except (RoutingError, ProviderError) as e:
                log(f"      accent check skipped for {u.id}: {str(e)[:120]}")
                return None

        check, vec, lid = await asyncio.gather(backcheck(), embed(), language())
        # what a clip this short can hold (Whisper invents text on sub-second clips)
        heard = qa_ops.plausible(check.words, clip16.dur_ms, text, tgt_lp.speech_unit,
                                 tgt_lp.joiner)  # fmt: skip
        res = qa_ops.judge(
            text,
            heard,
            tgt_lp.joiner,
            vec,
            ref.centroid if ref else None,
            ref.self_sim if ref else None,
            limits,
            speaker=u.speaker,
            centroids={r.speaker: r.centroid for r in refs.refs if r.centroid},
            dur_ms=clip16.dur_ms,
        )
        if lid is not None and base_lang(lid.lang) == base_lang(src_lang):
            res.mark_foreign(lid.lang)
        return res, heard

    # LLM in parallel with TTS (M5): as soon as a window is translated, its lines are voiced
    # ahead (take 0, trimmed, QA'd when the profile judges takes inline). The real voicing
    # pass below makes exactly the same calls, so it finds them in the cache; only lines whose
    # translation changes later (forms of address) are voiced again.
    def predict(text: str) -> int:
        return predict_ms(text, tgt_lp.speech_unit, tgt_lp.units_per_s,
                          tgt_lp.short_boost, tgt_lp.short_units)  # fmt: skip

    def choose(u: Unit, tr: Translation) -> tuple[str, str, int]:
        if u.id in ue.text:
            return ("edit", ue.text[u.id], predict(ue.text[u.id]))
        return pick_variant(tr, u.slot_ms, predict, t.variant_window, t.rate_p95)

    def unit_facts(u: Unit) -> dict[str, Any]:
        return {"unit.slot_ms": u.slot_ms, "unit.words": u.n_words, "unit.asr_conf": u.asr_conf}

    def line_facts(u: Unit, text: str) -> dict[str, Any]:
        """The unit's facts and the length of the line to voice (routes by line length)."""
        return {**unit_facts(u), "line.units": count_units(text, tgt_lp.speech_unit)}

    def synth_request(u: Unit, text: str, pred: int, seed: int, m: ProviderManifest) -> SynthReq:
        ref = refs.for_speaker(u.speaker)
        exact = m.caps.duration_control == "exact"
        return SynthReq(
            unit_id=u.id,
            # composed form: a decomposed Vietnamese tone mark is a separate token to a TTS
            text=unicodedata.normalize("NFC", text),
            lang=spec.tgt_lang,
            speaker=u.speaker,
            seed=seed,
            voice_key=ref.audio_key if ref else None,
            voice_text=unicodedata.normalize("NFC", ref.text) if ref and ref.text else None,
            # the voice's own silence comes on top, so its speech fills the planned length
            target_ms=exact_target_ms(u.slot_ms, pred, t.safe_ratio) + tts_pad.get(u.speaker, 0)
            if exact
            else None,
        )

    def engine_exact(u: Unit) -> bool:
        """Does the TTS this line routes to take an exact duration (OmniVoice)?"""
        try:
            pid = cx.router.route("tts.synthesize", {**cx.facts, **unit_facts(u)}).chain[0]
        except Exception:  # noqa: BLE001 - no runnable engine: the voicing pass reports it
            return False
        return cfg.providers[pid].caps.duration_control == "exact"

    def linear(text: str) -> int:
        """The line at the average rate: what speed-ups are measured against."""
        return predict_ms(text, tgt_lp.speech_unit, tgt_lp.units_per_s)

    def planned_fit(u: Unit, pred: int, prev, nxt, text: str | None = None) -> Fit:
        """The fit of an exact-duration take, known before synthesis: its length is planned."""
        return fit_unit(
            slot_start=u.start_ms, slot_end=u.end_ms, natural_ms=pred,
            clip_ms=exact_target_ms(u.slot_ms, pred, t.safe_ratio), prev=prev, next_=nxt, t=t,
            media_end_ms=media.duration_ms, rate_ref_ms=linear(text) if text else None,
        )  # fmt: skip

    def will_rewrite(u: Unit, text: str, pred: int) -> bool:
        """Sure to be rewritten (final): even with all the room around it, too fast. Voicing
        such a line ahead would be wasted."""
        return (
            profile.llm_rewrite
            and not ue.is_protected(u.id)
            and count_units(text, tgt_lp.speech_unit) > NO_REWRITE_UNITS
            and engine_exact(u)
            and planned_fit(u, pred, None, None, text).rate > t.rate_p95
        )

    voiced_by_id = {u.id: u for u in voiced}
    prefetched: dict[str, asyncio.Task] = {}
    prefetch_sem = asyncio.Semaphore(cfg.dub.throughput.units_in_flight)
    inline_qa = profile.takes > 1 and profile.backcheck

    async def prefetch(u: Unit, tr: Translation) -> None:
        async with prefetch_sem:
            try:
                _variant, text, pred = choose(u, ue.apply_to_translation(tr))
                if will_rewrite(u, text, pred) or u.id in ue.original:
                    return
                seed = ue.seed.get(u.id, 0)
                take: SynthRes = await cx.call(
                    "tts.synthesize",
                    lambda m: synth_request(u, text, pred, seed, m),
                    facts=line_facts(u, text),
                    subject=u.id,
                )
                trimmed: TrimRes = await cx.local(
                    "audio.trim", media_ops.trim, TrimReq(audio_key=take.audio_key), TrimRes,
                    subject=u.id,
                )  # fmt: skip
                if inline_qa:
                    await qa_take(u, trimmed.audio_key, text)
            except Exception as e:  # noqa: BLE001 - the voicing pass redoes it and reports
                log(f"      voicing ahead of {u.id} failed: {e}")

    async def translate_window(k: int, win: list[Unit], feedback: dict | None = None) -> None:
        i = k * size
        before = base_units[max(0, i - ctx_n) : i]
        after = base_units[i + size : i + size + ctx_n]
        req = TranslateReq(
            src_lang=src_lang,
            tgt_lang=spec.tgt_lang,
            units=win,
            before=before,
            after=after,
            context=sheet.subset_for(before + win + after),
            speech_unit=tgt_lp.speech_unit,
            units_per_s=tgt_lp.units_per_s,
            feedback={u.id: feedback[u.id] for u in win if feedback and u.id in feedback},
            wordless=[u.id for u in win if u.id in cries_set],
        )
        batch: TranslationBatch = await cx.call("mt.translate", req, subject=f"w{k:04d}")
        for tr in batch.items:
            translations[tr.unit_id] = tr
            u_v = voiced_by_id.get(tr.unit_id)
            if u_v is not None and not spec.subs_only:
                prefetched[tr.unit_id] = asyncio.create_task(prefetch(u_v, tr))

    await asyncio.gather(*(translate_window(k, w) for k, w in sorted(groups.items())))

    # Forms of address: rule check per line + whole-script consistency; the final profile
    # translates the windows with problems again, telling the translator what was wrong.
    generic = {g.lower() for g in tgt_lp.address_generic}

    foreign = _foreign_script(spec.tgt_lang)

    def address_issues() -> dict[str, str]:
        found: dict[str, str] = {}
        for u in units:  # a TTS reads leftover source characters aloud (M8)
            tr = translations.get(u.id)
            if tr and u.id not in ue.text and foreign:
                text = " ".join(v or "" for v in (tr.target, tr.short, tr.long))
                bad = sorted(set(foreign.findall(text)))
                if bad:
                    found[u.id] = (f"untranslated characters {''.join(bad[:8])}: write every word "
                                   f"in {spec.tgt_lang}, names transliterated")  # fmt: skip
        for u in units:
            tr = translations.get(u.id)
            if tr and u.id not in ue.text:
                # every variant: the timing fit may pick the short or the long one
                probs = []
                for variant in dict.fromkeys((tr.target, tr.short, tr.long)):
                    for why in address_ops.check_line(
                        u, variant, tr, sheet.cast, spec.tgt_lang, generic
                    ):
                        if why not in probs:
                            probs.append(why)
                if probs:
                    why = "; ".join(probs)
                    found[u.id] = f"{found[u.id]}; {why}" if u.id in found else why
        for uid, why in address_ops.listener_issues(units, translations).items():
            if uid not in ue.text:
                found[uid] = f"{found[uid]}; {why}" if uid in found else why
        for uid, why in address_ops.consistency(units, translations, sheet.cast).items():
            if uid not in ue.text:
                found[uid] = f"{found[uid]}; {why}" if uid in found else why
        return found

    addr = address_issues()
    for _round in range(2 if profile.llm_rewrite else 0):
        if not addr:
            break
        log(f"      forms of address: {len(addr)} lines to fix, translating their windows again")
        todo = {k: w for k, w in groups.items() if any(u.id in addr for u in w)}
        await asyncio.gather(*(translate_window(k, w, addr) for k, w in sorted(todo.items())))
        addr = address_issues()
    untranslated = {u.id for u in units if u.id not in translations}
    for u in units:  # never crash on a window the LLM answered partially: flag it instead
        if u.id in untranslated:
            translations[u.id] = Translation(unit_id=u.id, short=u.text, target=u.text, long=u.text)
    translations = {uid: ue.apply_to_translation(tr) for uid, tr in translations.items()}

    choices = {u.id: choose(u, translations[u.id]) for u in voiced}
    units = voiced
    # lines kept in the original audio cover their real sound (a song the ASR squeezed into
    # 300 ms went silent): their cues and the vocal sound around them, up to the dubbed lines
    extents = nonverbal_ops.original_extents(
        [(u.id, u.start_ms, u.end_ms, u.text) for u in units], set(ue.original),
        [(c.start_ms, c.end_ms, c.text) for c in (spec.hardsubs if hardsub_stats else [])],
        speech.regions,
    )  # fmt: skip

    placements: list[Placement] = []
    checks: dict[str, WordTimeline] = {}  # unit id -> back-check words (relative to the clip)
    qas: dict[str, qa_ops.QaResult] = {}
    trims: dict[str, TrimRes] = {}
    levels: dict[str, LineLevel] = {}  # unit id -> its level and the original's (mix gains)
    by_id = {u.id: u for u in units}
    backchecks: list[asyncio.Task] = []

    async def backcheck(p: Placement) -> None:
        """QA of the final clip when it was not checked take by take (one-take profiles)."""
        res, words_p = await qa_take(by_id[p.unit_id], p.audio_key, p.text)
        qas[p.unit_id] = res
        checks[p.unit_id] = WordTimeline(lang=spec.tgt_lang, words=words_p)

    async def dub_and_check(i: int) -> Placement:
        p = await dub_unit(i)
        if profile.backcheck and p.unit_id not in qas and p.variant != "original":
            backchecks.append(asyncio.create_task(backcheck(p)))
        return p

    engine = media_ops.stretch_engine()
    if not spec.subs_only:
        stage(f"[6/7] voicing and fitting {len(units)} units")
        sem = asyncio.Semaphore(cfg.dub.throughput.units_in_flight)

        def neighbor(u: Unit, other: Unit, gap: int) -> Neighbor:
            return Neighbor(
                natural_ms=choices[other.id][2],
                slot_ms=other.slot_ms,
                gap_ms=max(0, gap),
                same_speaker=other.speaker == u.speaker,
            )

        async def dub_unit(i: int) -> Placement:
            u = units[i]
            prev = neighbor(u, units[i - 1], u.start_ms - units[i - 1].end_ms) if i > 0 else None
            nxt = (
                neighbor(u, units[i + 1], units[i + 1].start_ms - u.end_ms)
                if i + 1 < len(units)
                else None
            )
            variant, text, pred = choices[u.id]
            if u.id in ue.original:  # not voiced: the original vocals of the line, as they are
                a0, b0 = extents.get(u.id, (u.start_ms, u.end_ms))
                b0 = max(b0, a0 + 10)  # covered by the line before it: a 10 ms sliver
                orig: AudioRes = await cx.local(
                    "audio.cut", media_ops.cut,
                    CutReq(audio_key=stems.vocals_key, start_ms=a0, end_ms=b0),
                    AudioRes, subject=u.id,
                )  # fmt: skip
                # its subtitle: the full translation (no voice to fit into the slot)
                tr_o = translations.get(u.id)
                if tr_o and variant in VARIANT_ORDER and tr_o.target:
                    text = tr_o.target
                return Placement(
                    unit_id=u.id, speaker=u.speaker, slot_start_ms=u.start_ms,
                    slot_end_ms=u.end_ms, start_ms=a0, end_ms=a0 + orig.dur_ms,
                    natural_ms=orig.dur_ms, clip_ms=orig.dur_ms, rate=1.0, variant="original",
                    text=text, flags=["original"], audio_key=orig.audio_key,
                )  # fmt: skip
            rewrites = 0
            tr_u = translations.get(u.id)
            base_seed = ue.seed.get(u.id, 0)
            if u.id in prefetched:  # its first take may still be in flight: do not redo it
                await prefetched[u.id]
            qa: qa_ops.QaResult | None = None
            check_words: list[Word] = []

            async def rewrite(text: str, rate: float) -> str | None:
                """A shorter line from the LLM; None when it breaks the forms of address."""
                cur = count_units(text, tgt_lp.speech_unit)
                rw = await cx.call(
                    "mt.rewrite",
                    RewriteReq(
                        unit_id=u.id,
                        tgt_lang=spec.tgt_lang,
                        source_text=u.text,
                        text=text,
                        current_units=cur,
                        target_units=max(1, int(cur * t.rate_p95 / rate)),
                        self_term=tr_u.self_term if tr_u else None,
                        other_term=tr_u.other_term if tr_u else None,
                    ),
                    subject=u.id,
                )
                if tr_u and address_ops.check_line(
                    u, rw.text, tr_u, sheet.cast, spec.tgt_lang, generic
                ):
                    return None
                return rw.text

            async with sem:
                # Exact-duration engines: the fit is known before synthesis, so a line that is
                # too fast is rewritten first and voiced once.
                # a line of a few syllables is not shortened: one word less is worse to hear
                # than a line that runs a little into the pause
                short = count_units(text, tgt_lp.speech_unit) <= NO_REWRITE_UNITS
                rewritable = profile.llm_rewrite and not ue.is_protected(u.id) and not short
                if rewritable and engine_exact(u):
                    while rewrites < profile.max_rewrites:
                        planned = planned_fit(u, pred, prev, nxt, text)
                        if planned.rate <= t.rate_p95:
                            break
                        rewrites += 1
                        new = await rewrite(text, planned.rate)
                        if new is None:
                            break
                        text, variant, pred = new, "rewrite", predict(new)

                async def one_take(text: str, pred: int, k: int):
                    def make_req(
                        m: ProviderManifest, text=text, pred=pred, seed=base_seed + 1000 * k
                    ) -> SynthReq:
                        return synth_request(u, text, pred, seed, m)

                    take: SynthRes = await cx.call(
                        "tts.synthesize", make_req, facts=line_facts(u, text), subject=u.id
                    )
                    # Fit the voice, not the clip: silence the engine left around it would
                    # shift the onset and waste the slot.
                    take_trim: TrimRes = await cx.local(
                        "audio.trim",
                        media_ops.trim,
                        TrimReq(audio_key=take.audio_key),
                        TrimRes,
                        subject=u.id,
                    )
                    if not inline_qa:
                        return (None, take, take_trim, [])
                    res, words_k = await qa_take(u, take_trim.audio_key, text)
                    return (res, take, take_trim, words_k)

                async def voice(text: str, pred: int):
                    # Up to `takes` deterministic takes (seed, seed+1000, ...): keep the first
                    # that passes the QA gate, else the best one (flagged). Take 0 alone first
                    # (most lines pass on it); when it fails, the other takes run at once, not
                    # one after another: a hard line's retakes in series were the tail of the
                    # drama's voicing (13 rounds, ~120 s; M10).
                    takes = [await one_take(text, pred, 0)]
                    n = max(1, profile.takes) if inline_qa else 1
                    for k in range(1, n):
                        if any(b[0].passed for b in takes):
                            break
                        if cfg.dub.throughput.parallel_retakes:
                            takes += await asyncio.gather(
                                *(one_take(text, pred, j) for j in range(k, n))
                            )
                            break
                        takes.append(await one_take(text, pred, k))
                    passing = [b for b in takes if b[0] is not None and b[0].passed]
                    best = passing[0] if passing else takes[0]
                    if not passing:
                        for b in takes[1:]:  # the best score; the earlier take on a tie
                            if (
                                b[0] is not None
                                and best[0] is not None
                                and (b[0].score > best[0].score)
                            ):
                                best = b
                    lf = line_facts(u, text)
                    if inline_qa and best[0] is not None and best[0].foreign:
                        best = await engine_retry(u, text, pred, base_seed, lf, best)
                    if inline_qa and best[0] is not None and not best[0].passed:
                        best = await engine_retry(
                            u, text, pred, base_seed, lf, best, fact="take.qa_retry"
                        )
                    return best

                def fit_of(best, text: str, pred: int) -> Fit:
                    synth, trimmed = best[1], best[2]
                    return fit_unit(
                        slot_start=u.start_ms,
                        slot_end=u.end_ms,
                        natural_ms=pred if synth.exact else trimmed.dur_ms,
                        clip_ms=trimmed.dur_ms,
                        prev=prev,
                        next_=nxt,
                        t=t,
                        media_end_ms=media.duration_ms,
                        rate_ref_ms=linear(text) if synth.exact else None,
                    )

                async def measured_variant(text: str, variant: str, pred: int, best, fit):
                    """Variants voiced and measured, not only predicted: (1) an engine without
                    duration control (VieNeu) got the short variant on a prediction, and the
                    full line may fit after all ("Không hứng" for "Không hứng thú"); (2) every
                    take failed the QA gate: another wording may be heard right."""
                    cands: list[tuple[str, str]] = []
                    if not best[1].exact and variant == "short":
                        cands.append(("target", tr_u.target))
                    if inline_qa and best[0] is not None and not best[0].passed:
                        cands += [("target", tr_u.target), ("short", tr_u.short),
                                  ("long", tr_u.long)]  # fmt: skip
                    tried = {text}
                    alts: list[tuple[str, str]] = []
                    for name, alt in cands:
                        if alt and alt not in tried:
                            tried.add(alt)
                            alts.append((name, alt))
                    # every wording voiced at once (parallel_retakes), judged in order
                    if cfg.dub.throughput.parallel_retakes:
                        voiced_alts = await asyncio.gather(*(voice(a, predict(a)) for _, a in alts))
                    else:
                        voiced_alts = [None] * len(alts)
                    for i, (name, alt) in enumerate(alts):
                        alt_pred = predict(alt)
                        alt_best = voiced_alts[i] or await voice(alt, alt_pred)
                        alt_fit = fit_of(alt_best, alt, alt_pred)
                        fuller = VARIANT_ORDER[name] > VARIANT_ORDER[variant]
                        if _prefer_take(alt_best[0], alt_fit.rate, best[0], fit.rate,
                                        t.rate_p95, fuller):  # fmt: skip
                            text, variant, pred, best, fit = alt, name, alt_pred, alt_best, alt_fit
                        if best[0] is None or best[0].passed:
                            break
                    return text, variant, pred, best, fit

                while True:
                    best = await voice(text, pred)
                    fit = fit_of(best, text, pred)
                    if tr_u and variant in VARIANT_ORDER and not ue.is_protected(u.id):
                        text, variant, pred, best, fit = await measured_variant(
                            text, variant, pred, best, fit
                        )
                    qa, synth, trimmed, check_words = best
                    natural = pred if synth.exact else trimmed.dur_ms
                    if (
                        fit.rate > t.rate_p95
                        and count_units(text, tgt_lp.speech_unit) > NO_REWRITE_UNITS
                        and profile.llm_rewrite
                        and rewrites < profile.max_rewrites
                        and not ue.is_protected(u.id)
                    ):
                        rewrites += 1
                        # A shorter line must not break the forms of address: keep the old
                        # one (and stop rewriting) when the rewrite does.
                        new = await rewrite(text, fit.rate)
                        if new is None:
                            break
                        text, variant, pred = new, "rewrite", predict(new)
                        continue
                    break
                stretched: AudioRes = await cx.local(
                    "audio.stretch",
                    media_ops.stretch,
                    StretchReq(audio_key=trimmed.audio_key, rate=fit.stretch, engine=engine),
                    AudioRes,
                    subject=u.id,
                )
                trims[u.id] = trimmed
                if qa is not None:
                    qas[u.id] = qa
                    checks[u.id] = WordTimeline(
                        lang=spec.tgt_lang, words=_scale(check_words, 1 / max(fit.stretch, 1e-6))
                    )
            if cfg.dub.mix.match_level:  # its level and the original's (gains at the end)
                levels[u.id] = await cx.local(
                    "audio.line_level",
                    level_ops.line_level,
                    LineLevelReq(
                        vocals_key=stems.vocals_key, slot_start_ms=u.start_ms,
                        slot_end_ms=u.end_ms, audio_key=stretched.audio_key,
                    ),
                    LineLevel,
                    subject=u.id,
                )  # fmt: skip
            flags = list(fit.flags) + (["untranslated"] if u.id in untranslated else [])
            if u.id in addr:
                flags.append("address")
            if qa is not None and not qa.passed:
                flags.append("qa_failed")
            return Placement(
                unit_id=u.id,
                speaker=u.speaker,
                slot_start_ms=u.start_ms,
                slot_end_ms=u.end_ms,
                start_ms=fit.start_ms,
                end_ms=fit.start_ms + stretched.dur_ms,
                natural_ms=natural,
                clip_ms=trimmed.dur_ms,
                rate=fit.rate,
                variant=variant,
                text=text,
                rewrites=rewrites,
                flags=flags,
                audio_key=stretched.audio_key,
            )

        placements = list(await asyncio.gather(*(dub_and_check(i) for i in range(len(units)))))
        # voicing ahead of translations that were replaced later: let them finish quietly
        await asyncio.gather(*prefetched.values(), return_exceptions=True)

    # Post-hoc checks (one-take profiles) started as each line was placed; they finish while
    # the tiles are mixed, and feed subtitles, risk and the report.
    checking = asyncio.gather(*backchecks)

    # ---------------------------------------------------------------- mix + subtitles
    stage("[7/7] mixing and writing subtitles")
    dub_audio_key = None
    dub_m4a_key = None
    tiles: list[Tile] = []
    level_stats: dict[str, Any] = {}
    if placements and levels:
        gains = level_ops.gains_for(
            levels,
            {p.unit_id: p.speaker for p in placements},
            cfg.dub.mix.level_max_dev_db,
            cfg.dub.mix.level_max_gain_db,
        )
        placements = [
            p.model_copy(update={"gain_db": gains.get(p.unit_id, 0.0)}) for p in placements
        ]
        level_stats = level_ops.level_metrics(levels, gains)
    export: asyncio.Task | None = None
    nonverbal: list[Placement] = []
    nonverbal_stats: dict[str, Any] = {}
    if placements and cfg.dub.mix.keep_nonverbal:
        nonverbal, nv_stats = await _nonverbal_events(
            cx, cfg, speech.regions, voiced,
            spec.hardsubs if hardsub_stats else [], stems.vocals_key, vocals16.audio_key,
            src_lang, src_lp, log, kept=list(extents.values()),
        )  # fmt: skip
        nonverbal_stats = nv_stats
    if placements:
        mixed = placements + nonverbal
        tile_reqs = [
            TileReq(
                bed_key=stems.instrumental_key,
                start_ms=a,
                end_ms=b,
                placements=[p for p in mixed if p.start_ms < b and p.end_ms > a],
                bed_gain_db=cfg.dub.mix.bed_gain_db,
            )
            for a, b in mix_ops.plan_tiles(media.duration_ms)
        ]
        tiles = list(
            await asyncio.gather(
                *(
                    cx.local(
                        "audio.mix_tile",
                        mix_ops.mix_tile,
                        r,
                        Tile,
                        subject=f"t{r.start_ms // 1000:05d}",
                    )
                    for r in tile_reqs
                )
            )
        )
        if spec.export:  # in the background: the subtitles are written meanwhile

            async def export_audio(keys: list[str]) -> tuple[str, str]:
                full: AudioRes = await cx.local(
                    "audio.concat", mix_ops.concat, ConcatReq(keys=keys), AudioRes
                )
                m4a: AudioRes = await cx.local(
                    "audio.master",
                    mix_ops.master,
                    MasterReq(audio_key=full.audio_key, loudness_lufs=cfg.dub.mix.loudness_lufs),
                    AudioRes,
                )
                return full.audio_key, m4a.audio_key

            export = asyncio.create_task(export_audio([x.audio_key for x in tiles]))
        await checking
        tgt_items = [
            SubItem(
                start_ms=p.start_ms,
                end_ms=p.end_ms,
                text=p.text,
                words=_shift(checks[p.unit_id].words, p.start_ms) if p.unit_id in checks else [],
            )
            for p in placements
            if p.unit_id not in cries_set  # a cry kept as it is needs no subtitle
        ]
    else:
        await checking
        tgt_items = [
            SubItem(start_ms=u.start_ms, end_ms=u.end_ms, text=choices[u.id][1]) for u in units
        ]

    async def subs_for(lang: str, items: list[SubItem]) -> SubtitleTrack:
        lp = cfg.langpack(lang)
        req = SubsReq(
            lang=lang, items=items, cpl=lp.subs.cpl, cps=lp.subs.cps, max_lines=lp.subs.max_lines
        )
        return await cx.local("subs.render", subs_ops.render, req, SubtitleTrack, subject=lang)

    src_items = [
        SubItem(
            start_ms=u.start_ms,
            end_ms=u.end_ms,
            text=u.text,
            words=[
                w for w in worded if w.start_ms >= u.start_ms - 50 and w.end_ms <= u.end_ms + 50
            ],
        )
        for u in units
    ]
    tgt_track, src_track = await asyncio.gather(
        subs_for(spec.tgt_lang, tgt_items), subs_for(src_lang, src_items)
    )
    if export is not None:
        dub_audio_key, dub_m4a_key = await export

    # ---------------------------------------------------------------- outputs for review
    backcheck_text = {uid: join_words(c.words, tgt_lp.joiner) for uid, c in checks.items()}
    backcheck_cer = {
        p.unit_id: round(cer(p.text, backcheck_text[p.unit_id]), 3)
        for p in placements
        if p.unit_id in backcheck_text
    }
    metrics = sync_metrics(
        units,
        placements,
        trims,
        backcheck_cer,
        tgt_track,
        tgt_lp,
        spec.tgt_lang,
        seg,
        media.duration_ms,
        tiles,
    )
    metrics.update(speaker_metrics(diar, purity, qas, refs))
    metrics.update(level_stats)
    metrics.update(hardsub_stats)
    metrics.update(nonverbal_stats)
    metrics.update({f"speakers_{k}": v for k, v in speaker_stats.items()
                    if k not in ("names", "kinds", "auto_original", "unsure_spans")})  # fmt: skip
    if "auto_original" in speaker_stats:
        metrics["auto_original"] = speaker_stats["auto_original"]
    if auditions:  # how much the audition changed the voices (see _audition_voices)
        metrics["audition_switched"] = sum(1 for a in auditions if a.chosen)
        gains = [
            a.sims[a.chosen] - a.sims[0]
            for a in auditions
            if a.sims[a.chosen] is not None and a.sims[0] is not None
        ]
        if gains:
            metrics["audition_sim_gain"] = round(sum(gains) / len(gains), 4)
    metrics["address_flagged"] = len(addr)
    result = JobResult(
        job_id=job_id,
        job_dir=job_dir,
        src_lang=src_lang,
        tgt_lang=spec.tgt_lang,
        dub_audio_key=dub_audio_key,
        dub_m4a_key=dub_m4a_key,
        subs={spec.tgt_lang: tgt_track.srt_key, f"{src_lang}.src": src_track.srt_key},
        placements=placements,
        tiles=tiles,
        a48k_key=media.a48k_key,
        warnings=warnings,
        redone=dict(Counter(st.op for st in cx.stats if not st.hit)),
        metrics=metrics,
        calls=len(cx.stats),
        cache_hits=sum(1 for s in cx.stats if s.hit),
    )
    summary = _write_outputs(
        job_dir,
        cx,
        tracer,
        seg,
        units,
        ue,
        sheet,
        translations,
        placements,
        tiles,
        tgt_track,
        tgt_lp,
        metrics,
        backcheck_cer,
        backcheck_text,
        {uid: q.as_dict() for uid, q in qas.items()},
        purity,
        addr,
        _unsure_units(units, speaker_stats.get("unsure_spans", [])),
    )
    log(f"      {summary}")
    # the voice references used (clip key, transcript): review, experiments, re-voicing
    (job_dir / "refs.json").write_text(refs.model_dump_json(indent=1), encoding="utf-8")
    cx.flush_stats()
    ws.db.upsert_job(
        job_id,
        source=spec.source_name,
        src_lang=src_lang,
        tgt_lang=spec.tgt_lang,
        profile=profile_name,
        status="done",
    )
    return result


# -------------------------------------------------------------------- report


def _analysis_result(cx, tracer, ws, job_id, job_dir, spec, seg, src_lang, media) -> JobResult:
    """Phase A only: the segmentation and the routes that produced it."""
    tracer.save(job_dir / "trace.json")
    with open(job_dir / "routes.jsonl", "w", encoding="utf-8") as f:
        for r in cx.routes:
            f.write(r.model_dump_json() + "\n")
    (job_dir / "segmentation.json").write_text(seg.model_dump_json(indent=1), encoding="utf-8")
    lens = [u.slot_ms for u in seg.units]
    metrics = {
        "units": len(seg.units),
        "sentences": len(seg.sentences),
        "unit_s_max": round(max(lens) / 1000, 2) if lens else 0,
        "units_over_15s": sum(1 for x in lens if x > 15000),
        "units_under_1s": sum(1 for x in lens if x < 1000),
    }
    (job_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    cx.flush_stats()
    ws.db.upsert_job(
        job_id, source=spec.source_name, src_lang=src_lang, tgt_lang=spec.tgt_lang,
        profile="analysis", status="done",
    )  # fmt: skip
    return JobResult(
        job_id=job_id, job_dir=job_dir, src_lang=src_lang, tgt_lang=spec.tgt_lang,
        dub_audio_key=None, dub_m4a_key=None, subs={}, a48k_key=media.a48k_key,
        metrics=metrics, redone=dict(Counter(st.op for st in cx.stats if not st.hit)),
        calls=len(cx.stats), cache_hits=sum(1 for s in cx.stats if s.hit),
    )  # fmt: skip


def cast_markdown(
    sheet: ContextSheet, units: list[Unit], tgt_lang: str, voices: list | None = None
) -> str:
    """The cast sheet for a human to check: who is who (with talk time and first lines, to
    recognise them) and how each pair addresses each other in the target language."""
    talk: dict[str, list[Unit]] = {}
    for u in units:
        talk.setdefault(u.speaker, []).append(u)
    head = "| id | name | gender | age | role | voice says | talk | first lines |"
    out = ["# Cast sheet", "", head, "|---|---|---|---|---|---|---|---|"]
    rows = {s.id: s for s in sheet.cast.speakers}
    hint = {v.speaker: v for v in voices or []}
    for spk in sorted(set(rows) | set(talk)):
        s = rows.get(spk)
        us = talk.get(spk, [])
        secs = sum(u.end_ms - u.start_ms for u in us) / 1000
        first = " / ".join(u.text for u in us[:2]).replace("|", "/")
        cells = [spk, s.name or "" if s else "", s.gender if s else "", s.age_band if s else "",
                 (s.role or "").replace("|", "/") if s else ""]  # fmt: skip
        mark = " (confirmed)" if s and s.locked else ""
        v = hint.get(spk)
        said = f"{v.gender} ({v.gender_conf:.2f}), {v.age_band}" if v else ""
        talked = f"{secs:.0f} s, {len(us)} lines"
        out.append(f"| {' | '.join(cells)}{mark} | {said} | {talked} | {first[:120]} |")
    out += [
        "",
        f"## How they address each other ({tgt_lang})",
        "",
        "| speaker > listener | relation | self | other | since |",
        "|---|---|---|---|---|",
    ]
    for r in sheet.cast.relations:
        a = r.address.get(tgt_lang)
        out.append(
            f"| {r.speaker} > {r.listener} | {(r.relation or '').replace('|', '/')} | "
            f"{a.self_term if a else ''} | {a.other_term if a else ''} | "
            f"{r.since_ms / 1000:.0f} s |"
        )
    return "\n".join(out) + "\n"


def _cast_result(cx, tracer, ws, job_id, job_dir, spec, sheet, units, src_lang, media,
                 voices, profile_name) -> JobResult:  # fmt: skip
    """Stop at the cast sheet (`dub run --confirm-cast`): everything so far is cached, so the
    full run after the corrections starts at translation."""
    tracer.save(job_dir / "trace.json")
    (job_dir / "cast.json").write_text(sheet.model_dump_json(indent=2), encoding="utf-8")
    (job_dir / "cast.md").write_text(
        cast_markdown(sheet, units, spec.tgt_lang, voices), encoding="utf-8"
    )
    metrics = {"speakers": len(sheet.cast.speakers), "units": len(units)}
    (job_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    cx.flush_stats()
    ws.db.upsert_job(
        job_id, source=spec.source_name, src_lang=src_lang, tgt_lang=spec.tgt_lang,
        profile=profile_name, status="cast",
    )  # fmt: skip
    return JobResult(
        job_id=job_id, job_dir=job_dir, src_lang=src_lang, tgt_lang=spec.tgt_lang,
        dub_audio_key=None, dub_m4a_key=None, subs={}, a48k_key=media.a48k_key,
        metrics=metrics, redone=dict(Counter(st.op for st in cx.stats if not st.hit)),
        calls=len(cx.stats), cache_hits=sum(1 for s in cx.stats if s.hit),
    )  # fmt: skip


CALIB_MAX_PAD_MS = 800
# Diarization again when the cast sheet finds several people under one speaker id (M8: a
# TikTok drama of ~6 characters came out as 2 voices at 0.85; 0.75 gave 5).
FREE_PHASE_A_BELOW_MIB = 30000  # "auto": free phase-A models on GPUs under this
JOB_THREADS = 256  # threads a job may hold waiting on workers and CPU steps (M10)
REF_CANDIDATES = 3  # reference clips tried per speaker (M8)
NO_REWRITE_UNITS = 3  # lines this short are never shortened by the LLM (M8)
REF_TEXT_OK = 0.15  # a clip heard within this CER of its line's text is clean
VARIANT_ORDER = {"short": 0, "target": 1, "long": 2}  # the translator's wordings, by length


def _prefer_take(
    new: qa_ops.QaResult | None,
    new_rate: float,
    cur: qa_ops.QaResult | None,
    cur_rate: float,
    max_rate: float,
    fuller: bool,
) -> bool:
    """Is another wording's take better than the current one? It must fit (no faster than the
    timing allows, or than the current one); then a take heard right beats one that is not,
    the fuller wording wins between two good ones, the higher QA score between two bad ones."""
    if new_rate > max(max_rate, cur_rate):
        return False
    if new is None or cur is None:  # no QA gate: the fuller line when it fits
        return fuller
    if new.passed != cur.passed:
        return new.passed
    return fuller if new.passed else new.score > cur.score


DIAR_STEP = 0.1
DIAR_FLOOR = 0.7
DIAR_FINER_MIN_SHARE = 0.03


def _unsure_units(units: list[Unit], spans: list[tuple[int, int]]) -> set[str]:
    """Units holding most of a line whose speaker the attribution only guessed."""
    out = set()
    for s0, e0 in spans:
        for u in units:
            ov = min(u.end_ms, e0) - max(u.start_ms, s0)
            if ov > 0.5 * max(1, e0 - s0):
                out.add(u.id)
    return out


def _not_dialogue(
    units: list[Unit],
    cues: list[HardsubCue],
    hardsub_stats: dict[str, Any],
    speaker_stats: dict[str, Any],
) -> list[str]:
    """Lines kept in the original audio: songs, and voices that are not part of the speech.

    With lines attributed (text.speakers): a song is never dubbed; a background voice (a
    crowd, a TV) is not when no subtitle carries it; a person's line always is, subtitled or
    not ("没谁啊", an answer the subtitles left out, had stayed in Chinese). Without it: a line
    with no subtitle over it, when the subtitles carry the dialogue elsewhere.
    """
    uncaptioned: set[str] = set()
    if hardsub_stats:
        uncaptioned = set(hardsub_ops.uncaptioned(units, cues, hardsub_stats["hardsub_coverage"]))
    if not speaker_stats.get("applied"):
        return [u.id for u in units if u.id in uncaptioned]
    kinds = speaker_stats.get("kinds", {})
    return [
        u.id
        for u in units
        if kinds.get(u.speaker) == "singing"
        or (kinds.get(u.speaker) == "background" and u.id in uncaptioned)
    ]


def _turn_spans(turns: list[Turn], per_speaker: int = 5) -> tuple[list[tuple[int, int]], list[str]]:
    """The longest turns of each diarizer speaker (1.5-15 s), for the voice hints."""
    by: dict[str, list[Turn]] = {}
    for t in turns:
        if 1500 <= t.end_ms - t.start_ms <= 15000:
            by.setdefault(t.speaker, []).append(t)
    spans, speakers = [], []
    for spk, ts in sorted(by.items()):
        for t in sorted(ts, key=lambda x: x.start_ms - x.end_ms)[:per_speaker]:
            spans.append((t.start_ms, t.end_ms))
            speakers.append(spk)
    return spans, speakers


async def _nonverbal_events(
    cx: Ctx,
    cfg: Config,
    regions: list[tuple[int, int]],
    units: list[Unit],
    cues: list[HardsubCue],
    vocals_key: str,
    vocals16_key: str,
    src_lang: str,
    src_lp,
    log: Callable[[str], None],
    kept: list[tuple[int, int]] | None = None,
) -> tuple[list[Placement], dict[str, Any]]:
    """Laughter, cries, screams between the lines, kept in the original audio (see
    dub.ops.nonverbal): vocal sound where no line and no subtitle is, in which the source ASR
    hears no words (or only laughter and cries). A gap with words is dropped: it would leak the
    source language."""
    m = cfg.dub.mix
    busy = [(u.start_ms, u.end_ms) for u in units] + [(c.start_ms, c.end_ms) for c in cues]
    busy += kept or []  # lines kept original already cover their sound
    found = nonverbal_ops.gaps(regions, busy, pad_ms=m.nonverbal_pad_ms,
                               min_ms=m.nonverbal_min_ms)  # fmt: skip

    async def judge(span: tuple[int, int]) -> Placement | None:
        a, b = span
        clip: AudioRes = await cx.local(
            "audio.cut", media_ops.cut, CutReq(audio_key=vocals16_key, start_ms=a, end_ms=b),
            AudioRes, subject=f"nv{a:07d}",
        )  # fmt: skip
        try:
            heard: WordTimeline = await cx.call(
                "asr.transcribe", TranscribeReq(audio_key=clip.audio_key, lang=src_lang),
                subject=f"nv{a:07d}",
            )  # fmt: skip
        except (RoutingError, ProviderError) as e:
            log(f"      non-verbal sound at {a / 1000:.1f}s not checked: {str(e)[:120]}")
            return None
        text = join_words(heard.words, src_lp.joiner)
        if not nonverbal_ops.wordless(text, src_lp.is_interjection):
            return None
        orig: AudioRes = await cx.local(
            "audio.cut", media_ops.cut, CutReq(audio_key=vocals_key, start_ms=a, end_ms=b),
            AudioRes, subject=f"nv{a:07d}",
        )  # fmt: skip
        return Placement(
            unit_id=f"nv{a:07d}", speaker="-", slot_start_ms=a, slot_end_ms=b, start_ms=a,
            end_ms=a + orig.dur_ms, natural_ms=orig.dur_ms, clip_ms=orig.dur_ms, rate=1.0,
            variant="original", text=text, flags=["original", "nonverbal"],
            audio_key=orig.audio_key,
        )  # fmt: skip

    judged = await asyncio.gather(*(judge(s) for s in found))
    kept = [p for p in judged if p is not None]
    kept_s = sum(p.end_ms - p.start_ms for p in kept) / 1000
    if found:
        log(f"      non-verbal sounds kept: {len(kept)} of {len(found)} vocal gaps "
            f"({kept_s:.1f} s); {len(found) - len(kept)} held words and stay out")  # fmt: skip
    return kept, {"nonverbal_kept": len(kept), "nonverbal_kept_s": round(kept_s, 1),
                  "nonverbal_dropped": len(found) - len(kept)}  # fmt: skip


async def _attribute_speakers(
    cx: Ctx,
    diar: Diarization,
    words: list[Word],
    cues: list[HardsubCue] | None,
    src_lang: str,
    joiner: str,
    vocals16_key: str,
    log: Callable[[str], None],
) -> tuple[Diarization, dict[str, Any]]:
    """`text.speakers` on the lines of the dialogue, fused with the diarizer's voices
    (dub.ops.speakers). Returns the diarization to segment with, and stats for the report."""
    try:
        cx.router.route("text.speakers", dict(cx.facts))
    except RoutingError:
        return diar, {}
    voices: list[SpeakerVoice] = []
    spans, span_speakers = _turn_spans(diar.turns)
    if spans:
        try:
            prof: VoiceProfiles = await cx.call(
                "audio.voice_profile",
                VoiceProfileReq(audio_key=vocals16_key, spans=spans, speakers=span_speakers),
            )
            voices = prof.voices
        except (RoutingError, ProviderError) as e:
            log(f"      voice hints for the speakers skipped: {str(e)[:200]}")
    lines = speaker_ops.lines(words, diar.turns, cues, joiner)
    if len(lines) < 2:
        return diar, {}

    async def reading(k: int) -> Attribution | None:
        try:
            return await cx.call(
                "text.speakers", SpeakersReq(lang=src_lang, lines=lines, voices=voices, sample=k)
            )
        except (RoutingError, ProviderError) as e:
            log(f"      who says each line: reading {k + 1} failed ({str(e)[:200]})")
            return None

    n = max(1, cx.cfg.dub.speakers.votes)
    got = [a for a in await asyncio.gather(*(reading(k) for k in range(n))) if a is not None]
    got = [a for a in got if len(a.speaker) == len(lines)]
    if not got:
        # Not silent: on the drama the voices alone catch 4-24% of the speaker changes
        log(
            "      WARNING: who says each line could not be read; the speakers come from the "
            "voices alone, which merge similar voices (check them on the review page)"
        )
        return diar, {"failed": True}
    attr, agreement = speaker_ops.vote(got)
    turns, stats = speaker_ops.fuse(lines, attr, voices)
    stats["readings"] = len(got)
    stats["agreement"] = round(agreement, 3)
    if turns is None:
        n_bad, n = stats.get("gender_conflicts", 0), stats["lines"]
        log(f"      who says each line: not used ({stats.get('rejected')}: "
            f"{n_bad} of {n} lines would change gender)")  # fmt: skip
        return diar, stats
    people = ", ".join(f"{k} {v}" for k, v in stats["names"].items())
    log(f"      who says each line: {stats['characters']} people in {stats['lines']} lines, "
        f"{stats['moved']} lines moved off the voice clusters ({people}); "
        f"{len(got)} readings agree on {agreement:.0%} of the lines")  # fmt: skip
    speakers = sorted({t.speaker for t in turns}, key=lambda s: int(s[1:]))
    return Diarization(turns=turns, speakers=speakers), {**stats, "applied": True}


def _finer_diarization(cx: Ctx, sheet: ContextSheet, spec: JobSpec) -> float | None:
    """The next, finer clustering threshold when the cast sheet says one speaker id holds
    several people; None when there is nothing to do (or no finer step left)."""
    if spec.speakers_hint or spec.transcript_key:
        return None
    if not any(s.characters > 1 and not s.locked for s in sheet.cast.speakers):
        return None
    try:
        pid = cx.router.route("audio.diarize", dict(cx.facts)).chain[0]
    except Exception:  # noqa: BLE001 - no diarizer: nothing to refine
        return None
    if pid != "ecapa_cluster":
        return None
    cur = float(cx.params_for(cx.cfg.providers[pid], "audio.diarize").get("threshold", 0.85))
    nxt = round(cur - DIAR_STEP, 2)
    return nxt if nxt >= DIAR_FLOOR - 1e-9 else None


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


def _foreign_script(tgt_lang: str) -> re.Pattern | None:
    """Characters that never belong in a line of this target language (CJK in vi, en...)."""
    return None if base_lang(tgt_lang) in ("zh", "ja", "ko") else _CJK


async def _retranscribe_refs(
    cx: Ctx,
    refs: RefBank,
    lang: str,
    joiner: str,
    log: Callable[[str], None],
    keep_alternates: bool = True,
) -> RefBank:
    """Each reference's transcript, from the cut clip itself (M8), and the cleanest clip.

    OmniVoice is primed with the reference audio *and* its text; when they disagree (a unit's
    text for a clip whose edges moved, or several lines joined) the clone can speak part of
    the reference text, in the source language, inside the dub. The source ASR hears the
    clip again; an empty result keeps the old text.

    A clip the ASR hears far from its line's text holds something else too: a laugh, another
    voice, the edge of the next line. Every take of the master's lines of a zh drama began
    with two extra syllables ("sát nhến") from a reference clip with laughter (CER 0.31 between
    clip and line). So, of a speaker's candidate clips in rank order, the first heard within
    REF_TEXT_OK of its line is used, else the one heard closest."""

    async def one(ref: VoiceRef) -> VoiceRef:
        clip16: AudioRes = await cx.local(
            "audio.to_mono16k", media_ops.to_mono16k, AudioReq(audio_key=ref.audio_key),
            AudioRes, subject=f"ref-{ref.speaker}",
        )  # fmt: skip
        try:
            heard: WordTimeline = await cx.call(
                "asr.transcribe",
                TranscribeReq(audio_key=clip16.audio_key, lang=lang),
                subject=f"ref-{ref.speaker}",
            )
        except (RoutingError, ProviderError) as e:
            log(f"      reference of {ref.speaker}: transcript kept ({str(e)[:120]})")
            return ref
        text = join_words(heard.words, joiner).strip()
        return ref.model_copy(update={"text": text}) if text else ref

    olds = [*refs.refs, *refs.alternates]
    news = list(await asyncio.gather(*(one(r) for r in olds)))
    by: dict[str, list[tuple[float, VoiceRef]]] = {}
    for old, new in zip(olds, news, strict=True):  # refs first: rank order per speaker
        by.setdefault(new.speaker, []).append((cer(old.text, new.text), new))
    chosen: list[VoiceRef] = []
    rest: list[VoiceRef] = []
    for r in refs.refs:
        cands = by[r.speaker]
        ok = [i for i, (e, _) in enumerate(cands) if e <= REF_TEXT_OK]
        k = ok[0] if ok else min(range(len(cands)), key=lambda i: cands[i][0])
        err, best = cands[k]
        chosen.append(best)
        rest += [c for i, (_, c) in enumerate(cands) if i != k]
        if k or err > 0:
            which = f"clip {k + 1} of {len(cands)}" if len(cands) > 1 else "the clip"
            log(f"      reference of {r.speaker}: {which}, heard as its line within CER "
                f"{err:.2f}; its transcript is what the ASR heard")  # fmt: skip
    return RefBank(refs=chosen, alternates=rest if keep_alternates else [])


@dataclass
class Audition:
    speaker: str
    chosen: int  # index of the kept clip among the candidates (0 = the most typical one)
    sims: list[float | None]  # per candidate: mean similarity of its takes to the speaker
    cers: list[float | None]  # per candidate: mean back-check CER of its takes
    pad_ms: int  # silence added to every request for this voice (exact engines)


async def _audition_voices(
    cx: Ctx, refs: RefBank, lp: Langpack, lang: str, slack: float
) -> tuple[RefBank, dict[str, int], list[Audition]]:
    """Before voicing: which reference clip each speaker clones best, and how much silence an
    exact-duration TTS adds inside the length we ask for.

    Silence: OmniVoice fills the requested duration with speech plus some silence, mostly at
    the start; after trimming, the speech is squeezed (on a 60-minute speech: 90% of the
    requested length at p50, 76% for lines under 2 s, ~340 ms lead silence). The median
    silence trimmed off the kept clip's test takes is added to every later request for it.

    Audition: every candidate clip (the reference and RefBank.alternates) says the langpack's
    two calibration sentences; each take is back-checked (ASR) and embedded. Among the
    candidates whose takes are intelligible (mean CER within `slack` of the best), the one
    whose takes come out closest to the speaker's centroid is kept. Cached like any call, so
    runs stay reproducible.
    """
    if not lp.calibration:
        return refs, {}, []
    try:
        pid = cx.router.route("tts.synthesize", dict(cx.facts)).chain[0]
    except Exception:  # noqa: BLE001 - no TTS: nothing to audition
        return refs, {}, []
    exact = cx.cfg.providers[pid].caps.duration_control == "exact"
    cands: dict[str, list[VoiceRef]] = {}
    for r in [*refs.refs, *refs.alternates]:
        if r.audio_key:
            cands.setdefault(r.speaker, []).append(r)
    if not exact and all(len(c) < 2 for c in cands.values()):
        return refs, {}, []
    sentences = lp.calibration[:2]

    async def take(ref: VoiceRef, k: int, text: str, judge: bool):
        subject = f"calibration-{k}"
        pred = predict_ms(text, lp.speech_unit, lp.units_per_s)
        req = SynthReq(
            unit_id=f"calibration-{ref.speaker}-{k}", text=text, lang=lang, speaker=ref.speaker,
            seed=0, voice_key=ref.audio_key, voice_text=ref.text,
            target_ms=pred if exact else None,
        )  # fmt: skip
        synth: SynthRes = await cx.call("tts.synthesize", req, subject=subject)
        tr: TrimRes = await cx.local(
            "audio.trim", media_ops.trim, TrimReq(audio_key=synth.audio_key), TrimRes,
            subject=subject,
        )  # fmt: skip
        sim = err = None
        if judge:
            clip16: AudioRes = await cx.local(
                "audio.to_mono16k", media_ops.to_mono16k, AudioReq(audio_key=tr.audio_key),
                AudioRes, subject=subject,
            )  # fmt: skip
            try:
                chk: WordTimeline = await cx.call(
                    "asr.backcheck",
                    TranscribeReq(audio_key=clip16.audio_key, lang=lang, expected_text=text),
                    subject=subject,
                )
                err = cer(text, join_words(chk.words, lp.joiner))
            except (RoutingError, ProviderError):
                pass
            if ref.centroid:
                try:
                    e: Embeddings = await cx.call(
                        "audio.embed", EmbedReq(audio_key=clip16.audio_key), subject=subject
                    )
                    sim = refs_ops.cos_sim(e.vectors[0], ref.centroid)
                except (RoutingError, ProviderError):
                    pass
        return tr.lead_ms + tr.tail_ms, sim, err

    def mean(xs: list) -> float | None:
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    async def one(spk: str, rs: list[VoiceRef]) -> tuple[VoiceRef, Audition]:
        judge = len(rs) > 1
        out = await asyncio.gather(
            *(take(r, k, t, judge) for r in rs for k, t in enumerate(sentences))
        )
        n = len(sentences)
        rows = [out[i * n : (i + 1) * n] for i in range(len(rs))]
        sims = [mean([x[1] for x in row]) for row in rows]
        cers = [mean([x[2] for x in row]) for row in rows]
        chosen = 0
        if judge and any(x is not None for x in sims):
            known = [c for c in cers if c is not None]
            ok = [
                i for i, c in enumerate(cers) if not known or c is None or c <= min(known) + slack
            ]
            chosen = max(ok, key=lambda i: (sims[i] if sims[i] is not None else -1.0, -i))
        pad = min(CALIB_MAX_PAD_MS, int(statistics.median(x[0] for x in rows[chosen])))
        return rs[chosen], Audition(spk, chosen, sims, cers, pad if exact else 0)

    picked = await asyncio.gather(*(one(s, rs) for s, rs in cands.items()))
    kept = {ref.speaker: ref for ref, _ in picked}
    new_refs = [kept.get(r.speaker, r) for r in refs.refs]
    used = {r.audio_key for r in new_refs}
    others = [r for r in [*refs.refs, *refs.alternates] if r.audio_key not in used]
    pads = {a.speaker: a.pad_ms for _, a in picked} if exact else {}
    return RefBank(refs=new_refs, alternates=others), pads, [a for _, a in picked]


def _voice_spans(
    units: list[Unit], purity: dict[str, float], per_speaker: int = 5
) -> tuple[list[tuple[int, int]], list[str]]:
    """The longest clean lines of each speaker (one voice, 2-15 s), for the voice hints."""
    by: dict[str, list[Unit]] = {}
    for u in units:
        if purity.get(u.id, 1.0) >= 0.9 and 2000 <= u.slot_ms <= 15000:
            by.setdefault(u.speaker, []).append(u)
    spans, speakers = [], []
    for spk, us in sorted(by.items()):
        for u in sorted(us, key=lambda x: -x.slot_ms)[:per_speaker]:
            spans.append((u.start_ms, u.end_ms))
            speakers.append(spk)
    return spans, speakers


def _pauses(words: list[Word]) -> list[int]:
    """Silence after each word, up to the next one (0 after the last)."""
    return [max(0, b.start_ms - a.end_ms) for a, b in zip(words, words[1:], strict=False)] + [0]


def _wants_pauses(cx: Ctx, op: str) -> bool:
    """Only providers that read pauses get them: SaT's call keys (and cache) stay as they were."""
    try:
        pid = cx.router.route(op, dict(cx.facts)).chain[0]
    except Exception:  # noqa: BLE001 - routing errors surface at the call itself
        return False
    m = cx.cfg.providers[pid]
    return bool(cx.params_for(m, op).get("reads_pauses"))  # --param can switch it off


def _free_phase_a(cfg: Config) -> bool:
    v = cfg.dub.throughput.free_phase_a_models
    if v == "auto":
        mem = os.environ.get("DUB_GPU_MEM_MIB")  # set by dubd from nvidia-smi
        # 24 GB cards only: the 5-min drama peaks at 16 GB on a 32 GB card, where freeing cost
        # a 3.5 s stall on the critical path and restores later
        return bool(mem) and int(mem) < FREE_PHASE_A_BELOW_MIB
    return bool(v)


def _scale(words: list[Word], factor: float) -> list[Word]:
    """Word times of a take, after the take was time-stretched by 1/factor."""
    return [
        w.model_copy(
            update={"start_ms": round(w.start_ms * factor), "end_ms": round(w.end_ms * factor)}
        )
        for w in words
    ]


def speaker_metrics(diar, purity, qas, refs) -> dict[str, Any]:
    """Speakers found, line purity, and the QA gate's verdicts."""
    m: dict[str, Any] = {"speakers": len(diar.speakers)}
    vals = list(purity.values())
    if vals:
        m["units_impure"] = sum(1 for v in vals if v < 0.8)
        m["purity_p10"] = round(sorted(vals)[len(vals) // 10], 3)
    if qas:
        ratios = sorted(q.sim_ratio for q in qas.values() if q.sim_ratio is not None)
        m["qa_passed"] = round(sum(1 for q in qas.values() if q.passed) / len(qas), 3)
        m["qa_truncated"] = sum(1 for q in qas.values() if any("ending" in r for r in q.reasons))
        if ratios:
            m["voice_sim_ratio_p50"] = round(ratios[len(ratios) // 2], 3)
            m["voice_sim_ratio_p10"] = round(ratios[len(ratios) // 10], 3)
    m["ref_sim"] = {r.speaker: r.sim for r in refs.refs}
    return m


def _shift(words: list[Word], offset_ms: int) -> list[Word]:
    return [
        w.model_copy(update={"start_ms": w.start_ms + offset_ms, "end_ms": w.end_ms + offset_ms})
        for w in words
    ]


def sync_metrics(
    units, placements, trims, backcheck_cer, tgt_track, tgt_lp, tgt_lang, seg, media_ms, tiles
) -> dict[str, Any]:
    """The numbers behind priorities 1 and 2 (segmentation, timing) and the subtitles.

    onset error: |where the dubbed voice starts - where the original voice starts|. Clips are
    trimmed to the voice, so the dubbed onset is the placement start.
    """
    m: dict[str, Any] = {}
    lens = [u.slot_ms for u in seg.units]
    m["units"] = len(seg.units)
    m["unit_s_max"] = round(max(lens) / 1000, 2) if lens else 0
    m["units_over_15s"] = sum(1 for x in lens if x > 15000)
    m["units_under_1s"] = sum(1 for x in lens if x < 1000)
    if placements:
        by_id = {u.id: u for u in units}
        onset = [abs(p.start_ms - by_id[p.unit_id].start_ms) for p in placements]
        rates = [p.rate for p in placements]
        m["onset_err_ms_p50"] = round(statistics.median(onset))
        m["onset_err_ms_p95"] = round(_pct(onset, 0.95))
        m["speedup_p50"] = round(statistics.median(rates), 3)
        m["speedup_p95"] = round(_pct(rates, 0.95), 3)
        m["speedup_max"] = round(max(rates), 3)
        m["overflow"] = sum(1 for p in placements if "overflow" in p.flags)
        # How fast the dub really speaks (units per second of placed voice), against the
        # language's natural rate: the speed-up above compares with a prediction, this does not.
        spoken = [
            count_units(p.text, tgt_lp.speech_unit) / ((p.end_ms - p.start_ms) / 1000)
            for p in placements
            if p.end_ms - p.start_ms > 300
        ]
        if spoken:
            m["speech_rate_p50"] = round(statistics.median(spoken), 2)
            m["speech_rate_p90"] = round(_pct(spoken, 0.90), 2)
            m["speech_rate_natural"] = tgt_lp.units_per_s
        if trims:
            lead = [t.lead_ms for t in trims.values()]
            tail = [t.tail_ms for t in trims.values()]
            m["tts_silence_ms_p50"] = {
                "lead": round(statistics.median(lead)),
                "tail": round(statistics.median(tail)),
            }
    if backcheck_cer:
        vals = list(backcheck_cer.values())
        m["backcheck_cer_p50"] = round(statistics.median(vals), 3)
        m["backcheck_cer_p95"] = round(_pct(vals, 0.95), 3)
    if tiles:
        m["drift_ms"] = abs(tiles[-1].end_ms - media_ms)
    m["subs_cues"] = len(tgt_track.cues)
    m["subs_cps_violations"] = subs_ops.cps_violations(tgt_track.cues, tgt_lp.subs.cps, tgt_lang)
    return m


def _write_outputs(
    job_dir,
    cx: Ctx,
    tracer: Tracer,
    seg,
    units,
    ue,
    sheet,
    translations,
    placements,
    tiles,
    tgt_track,
    tgt_lp,
    metrics,
    backcheck_cer,
    backcheck_text,
    qa_rows,
    purity,
    addr,
    speaker_unsure: set[str] | None = None,
) -> str:
    tracer.save(job_dir / "trace.json")
    with open(job_dir / "routes.jsonl", "w", encoding="utf-8") as f:
        for r in cx.routes:
            f.write(r.model_dump_json() + "\n")
    (job_dir / "cast.json").write_text(sheet.model_dump_json(indent=2), encoding="utf-8")
    by_id = {p.unit_id: p for p in placements}
    rows = []
    for u in units:
        tr = translations.get(u.id)
        p = by_id.get(u.id)
        edited = u.id in ue.text or u.id in ue.seed or u.id in ue.speaker or u.id in ue.nudge
        risk, why = risk_ops.score(
            u, p, tr, edited=edited, cer=backcheck_cer.get(u.id), qa=qa_rows.get(u.id),
            purity=purity.get(u.id), address=addr.get(u.id),
            speaker_unsure=u.id in (speaker_unsure or set()),
        )  # fmt: skip
        rows.append(
            {
                "unit": u.model_dump(),
                "translation": tr.model_dump() if tr else None,
                "placement": p.model_dump() if p else None,
                "risk": risk,
                "risk_reasons": why,
                "edited": edited,
                "locked": ue.is_protected(u.id),
                "seed": ue.seed.get(u.id, 0),
                "backcheck_cer": backcheck_cer.get(u.id),
                "backcheck_text": backcheck_text.get(u.id),
                "qa": qa_rows.get(u.id),
                "purity": purity.get(u.id),
                "address_issue": addr.get(u.id),
            }
        )
    (job_dir / "tiles.json").write_text(
        json.dumps([t.model_dump() for t in tiles], indent=1), encoding="utf-8"
    )
    (job_dir / "units.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (job_dir / "segmentation.json").write_text(seg.model_dump_json(indent=1), encoding="utf-8")
    (job_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    report = render_report(cx, seg, sheet, placements, tgt_track, tgt_lp, metrics)
    risks = residual_risks(rows)
    gap = "\n\n"
    head, _, rest = report.partition(gap)  # the risks right under the title
    (job_dir / "report.md").write_text(head + gap + "\n".join(risks) + gap + rest, encoding="utf-8")
    return risk_summary(rows)


RISK_REVIEW = 3.0  # a line this risky is worth a listen


def _risk_kind(reason: str) -> str:
    """The family of a risk reason, for the summary."""
    for key, kind in (
        ("failed the QA gate", "failed QA"),
        ("CER", "words"),
        ("voice", "voice"),
        ("fills ", "short fill"),
        ("end of the line", "cut off"),
        ("who says it", "speaker guessed"),
        ("another voice", "mixed voices"),
        ("sped up", "fast"),
        ("does not fit", "overflow"),
        ("forms of address", "address"),
        ("not translated", "untranslated"),
    ):
        if key in reason:
            return kind
    return "other"  # fmt: skip


def risk_summary(rows: list[dict]) -> str:
    """One line for the end of a run: how many lines are worth a listen, and why."""
    risky = [r for r in rows if r["risk"] >= RISK_REVIEW and not r["edited"]]
    if not risky:
        return f"residual risks: none of {len(rows)} lines above {RISK_REVIEW:g}"
    # lines per kind (a line counts once per kind; minor reasons are not listed)
    kinds = Counter(k for r in risky for k in {_risk_kind(x) for x in r["risk_reasons"]}
                    if k != "other")  # fmt: skip
    why = ", ".join(f"{n} {k}" for k, n in kinds.most_common())
    return f"residual risks: {len(risky)} of {len(rows)} lines worth a listen ({why})"


def residual_risks(rows: list[dict], top: int = 15) -> list[str]:
    """The report's first section: what is left to check after a one-pass run (M8)."""
    out = ["## Residual risks", "", risk_summary(rows).capitalize() + ".", ""]
    risky = sorted((r for r in rows if r["risk"] >= RISK_REVIEW and not r["edited"]),
                   key=lambda r: -r["risk"])  # fmt: skip
    for r in risky[:top]:
        u = r["unit"]
        text = (r["placement"] or {}).get("text") or u["text"]
        out.append(f"- {u['start_ms'] / 1000:7.1f} s `{u['id']}` risk {r['risk']}: {text[:60]} "
                   f"({'; '.join(r['risk_reasons'])})")  # fmt: skip
    if len(risky) > top:
        out.append(f"- ... {len(risky) - top} more in units.json (the review page sorts by risk)")
    return out


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def render_report(cx: Ctx, seg, sheet, placements, tgt_track, tgt_lp, metrics=None) -> str:
    lines = [f"# Job report: {cx.job_id}", ""]
    hits = sum(1 for s in cx.stats if s.hit)
    lines += [
        f"- Calls: {len(cx.stats)} (cache hits: {hits})",
        f"- Sentences: {len(seg.sentences)}, dub units: {len(seg.units)}",
        "",
    ]

    lines += [
        "## Calls per op",
        "",
        "| op | provider | calls | cache hits | run time (s) |",
        "|---|---|---:|---:|---:|",
    ]
    agg: dict[tuple[str, str], list] = {}
    for s in cx.stats:
        a = agg.setdefault((s.op, s.provider), [0, 0, 0.0])
        a[0] += 1
        a[1] += int(s.hit)
        a[2] += s.dur_ms / 1000
    for (op, prov), (n, h, sec) in sorted(agg.items()):
        lines.append(f"| {op} | {prov} | {n} | {h} | {sec:.2f} |")

    lines += ["", "## Routing", ""]
    seen: Counter = Counter()
    reasons: dict[tuple[str, str], str] = {}
    for r in cx.routes:
        seen[(r.op, r.source, r.rule_id or "-", r.chain[0])] += 1
        for rej in r.rejected:
            reasons[(r.op, rej.provider)] = rej.reason
    lines += ["| op | source | rule | chosen | times |", "|---|---|---|---|---:|"]
    for (op, src, rule, chosen), n in sorted(seen.items()):
        lines.append(f"| {op} | {src} | {rule} | {chosen} | {n} |")
    if reasons:
        lines += ["", "Rejected candidates:", ""]
        lines += [f"- `{op}` / `{p}`: {why}" for (op, p), why in sorted(reasons.items())]

    if placements:
        rates = [p.rate for p in placements]
        flags = Counter(f for p in placements for f in p.flags)
        lines += [
            "",
            "## Timing",
            "",
            f"- Speed-up factor: p50 {statistics.median(rates):.2f}, "
            f"p95 {_pct(rates, 0.95):.2f}, max {max(rates):.2f}",
            f"- Variants used: {dict(Counter(p.variant for p in placements))}",
            f"- Flags: {dict(flags) if flags else 'none'}",
        ]
        worst = sorted(placements, key=lambda p: -p.rate)[:5]
        lines += ["", "Units to review first (highest speed-up):", ""]
        lines += [f"- `{p.unit_id}` x{p.rate:.2f} {p.flags}: {p.text}" for p in worst]

    if metrics:
        lines += ["", "## Sync and quality", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in metrics.items()]

    lines += ["", "## Cast sheet", ""]
    for s in sheet.cast.speakers:
        lines.append(
            f"- {s.id}: name={s.name}, gender={s.gender} ({s.conf:.2f}), "
            f"age={s.age_band}, role={s.role}"
        )
    for r in sheet.cast.relations:
        terms = {k: v.model_dump(exclude_none=True) for k, v in r.address.items()}
        lines.append(f"- {r.speaker} -> {r.listener}: {r.relation or ''} [{r.formality}] {terms}")
    if sheet.flags:
        lines += ["", "Needs confirmation:", ""] + [f"- {f}" for f in sheet.flags]

    bad = subs_ops.cps_violations(tgt_track.cues, tgt_lp.subs.cps, tgt_lp.lang)
    lines += ["", "## Subtitles", "", f"- {len(tgt_track.cues)} cues, {bad} read too fast", ""]
    return "\n".join(lines)
