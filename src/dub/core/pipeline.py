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
import json
import os
import re
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dub.contracts.v1 import (
    AudioReq,
    AudioRes,
    Boundaries,
    BoundaryReq,
    ConcatReq,
    ContextReq,
    ContextSheet,
    DecodeReq,
    Diarization,
    DiarizeReq,
    Edit,
    Embeddings,
    EmbedReq,
    MasterReq,
    MediaInfo,
    Placement,
    RefBank,
    RefsReq,
    RewriteReq,
    Segmentation,
    SegmentReq,
    SeparateReq,
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
    Unit,
    VadReq,
    Word,
    WordTimeline,
    join_words,
)
from dub.core.config import Config, ProviderManifest
from dub.core.ctx import Ctx
from dub.core.executor import Executor
from dub.core.router import Router
from dub.core.trace import Tracer
from dub.ops import address as address_ops
from dub.ops import diarize as diar_ops
from dub.ops import edits as edits_ops
from dub.ops import media as media_ops
from dub.ops import mix as mix_ops
from dub.ops import qa as qa_ops
from dub.ops import refs as refs_ops
from dub.ops import risk as risk_ops
from dub.ops import segment as segment_ops
from dub.ops import subs as subs_ops
from dub.ops import vad as vad_ops
from dub.ops.align import cer
from dub.ops.timing import (
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
) -> JobResult:
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
        ),
        Segmentation,
        with_cas=False,
    )
    base_units = seg.units
    if spec.analysis_only:
        n_words, n_sent = len(words.words), len(seg.sentences)
        log(f"      {n_words} words -> {n_sent} sentences -> {len(seg.units)} units")
        return _analysis_result(cx, tracer, ws, job_id, job_dir, spec, seg, src_lang, media)
    edits = edits_ops.effective(spec.edits)
    ue = edits_ops.UnitEdits.from_edits(edits)
    worded = segment_ops.assign_speakers(asr_words, diar.turns)
    units, warnings = edits_ops.apply_structure(base_units, worded, edits, src_lp.joiner)
    for w in warnings:
        log(f"      edit skipped: {w}")
    voiced = ue.apply_to_units(units)  # speaker and slot edits
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
            vocals_key=stems.vocals_key, units=voiced, unit_vectors=unit_vectors, purity=purity
        ),
        RefBank,
    )
    if edits:
        log(f"      {len(edits)} human edits applied")
    if _free_phase_a(cfg):  # 24 GB GPUs: room for the TTS batches of phase B
        freed = await cx.free_models(
            ["audio.separate", "asr.transcribe", "text.boundaries"],
            keep_ops=["asr.backcheck", "tts.synthesize", "audio.embed"],
        )
        if freed:
            log(f"      GPU memory freed: {', '.join(freed)}")
    log(f"      {len(words.words)} words -> {len(seg.sentences)} sentences -> {len(units)} units")

    # ---------------------------------------------------------------- Barrier: context pass
    stage("[4/7] reading the whole script: summary, glossary, cast sheet")
    sheet: ContextSheet = await cx.call(
        "mt.context",
        ContextReq(src_lang=src_lang, tgt_lang=spec.tgt_lang, units=base_units, voices=diar.voices),
    )
    sheet = edits_ops.apply_cast(sheet, edits)  # the user's corrections win
    if spec.confirm_cast:
        return _cast_result(cx, tracer, ws, job_id, job_dir, spec, sheet, voiced, src_lang, media,
                            profile_name)  # fmt: skip

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
        voice_min_ms=q.voice_min_ms,
    )

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
        check: WordTimeline = await cx.call(
            "asr.backcheck",
            TranscribeReq(audio_key=clip16.audio_key, lang=spec.tgt_lang, expected_text=text),
            subject=u.id,
        )
        vec = None
        if ref and ref.centroid:
            e: Embeddings = await cx.call(
                "audio.embed", EmbedReq(audio_key=clip16.audio_key), subject=u.id
            )
            vec = e.vectors[0]
        res = qa_ops.judge(
            text,
            check.words,
            tgt_lp.joiner,
            vec,
            ref.centroid if ref else None,
            ref.self_sim if ref else None,
            limits,
            speaker=u.speaker,
            centroids={r.speaker: r.centroid for r in refs.refs if r.centroid},
            dur_ms=clip16.dur_ms,
        )
        return res, check.words

    # LLM in parallel with TTS (M5): as soon as a window is translated, its lines are voiced
    # ahead (take 0, trimmed, QA'd when the profile judges takes inline). The real voicing
    # pass below makes exactly the same calls, so it finds them in the cache; only lines whose
    # translation changes later (forms of address) are voiced again.
    def predict(text: str) -> int:
        return predict_ms(text, tgt_lp.speech_unit, tgt_lp.units_per_s)

    def choose(u: Unit, tr: Translation) -> tuple[str, str, int]:
        if u.id in ue.text:
            return ("edit", ue.text[u.id], predict(ue.text[u.id]))
        return pick_variant(tr, u.slot_ms, predict, t.variant_window)

    def unit_facts(u: Unit) -> dict[str, Any]:
        return {"unit.slot_ms": u.slot_ms, "unit.words": u.n_words, "unit.asr_conf": u.asr_conf}

    def synth_request(u: Unit, text: str, pred: int, seed: int, m: ProviderManifest) -> SynthReq:
        ref = refs.for_speaker(u.speaker)
        exact = m.caps.duration_control == "exact"
        return SynthReq(
            unit_id=u.id,
            text=text,
            lang=spec.tgt_lang,
            speaker=u.speaker,
            seed=seed,
            voice_key=ref.audio_key if ref else None,
            voice_text=ref.text if ref else None,
            target_ms=exact_target_ms(u.slot_ms, pred, t.safe_ratio) if exact else None,
        )

    voiced_by_id = {u.id: u for u in voiced}
    prefetched: dict[str, asyncio.Task] = {}
    prefetch_sem = asyncio.Semaphore(cfg.dub.throughput.units_in_flight)
    inline_qa = profile.takes > 1 and profile.backcheck

    async def prefetch(u: Unit, tr: Translation) -> None:
        async with prefetch_sem:
            try:
                _variant, text, pred = choose(u, ue.apply_to_translation(tr))
                seed = ue.seed.get(u.id, 0)
                take: SynthRes = await cx.call(
                    "tts.synthesize",
                    lambda m: synth_request(u, text, pred, seed, m),
                    facts=unit_facts(u),
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

    def address_issues() -> dict[str, str]:
        found: dict[str, str] = {}
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
                    found[u.id] = "; ".join(probs)
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

    placements: list[Placement] = []
    checks: dict[str, WordTimeline] = {}  # unit id -> back-check words (relative to the clip)
    qas: dict[str, qa_ops.QaResult] = {}
    trims: dict[str, TrimRes] = {}
    by_id = {u.id: u for u in units}
    backchecks: list[asyncio.Task] = []

    async def backcheck(p: Placement) -> None:
        """QA of the final clip when it was not checked take by take (one-take profiles)."""
        res, words_p = await qa_take(by_id[p.unit_id], p.audio_key, p.text)
        qas[p.unit_id] = res
        checks[p.unit_id] = WordTimeline(lang=spec.tgt_lang, words=words_p)

    async def dub_and_check(i: int) -> Placement:
        p = await dub_unit(i)
        if profile.backcheck and p.unit_id not in qas:
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
            facts = unit_facts(u)
            rewrites = 0
            tr_u = translations.get(u.id)
            base_seed = ue.seed.get(u.id, 0)
            if u.id in prefetched:  # its first take may still be in flight: do not redo it
                await prefetched[u.id]
            qa: qa_ops.QaResult | None = None
            check_words: list[Word] = []
            async with sem:
                while True:
                    # Up to `takes` deterministic takes (seed, seed+1000, ...): keep the first
                    # that passes the QA gate, else the best one (flagged).
                    best = None
                    for k in range(max(1, profile.takes) if inline_qa else 1):

                        def make_req(
                            m: ProviderManifest, text=text, pred=pred, seed=base_seed + 1000 * k
                        ) -> SynthReq:
                            return synth_request(u, text, pred, seed, m)

                        take: SynthRes = await cx.call(
                            "tts.synthesize", make_req, facts=facts, subject=u.id
                        )
                        # Fit the voice, not the clip: silence the engine left around it
                        # would shift the onset and waste the slot.
                        take_trim: TrimRes = await cx.local(
                            "audio.trim",
                            media_ops.trim,
                            TrimReq(audio_key=take.audio_key),
                            TrimRes,
                            subject=u.id,
                        )
                        if not inline_qa:
                            best = (None, take, take_trim, [])
                            break
                        res, words_k = await qa_take(u, take_trim.audio_key, text)
                        if best is None or res.score > best[0].score:
                            best = (res, take, take_trim, words_k)
                        if res.passed:
                            break
                    qa, synth, trimmed, check_words = best
                    natural = pred if synth.exact else trimmed.dur_ms
                    fit = fit_unit(
                        slot_start=u.start_ms,
                        slot_end=u.end_ms,
                        natural_ms=natural,
                        clip_ms=trimmed.dur_ms,
                        prev=prev,
                        next_=nxt,
                        t=t,
                        media_end_ms=media.duration_ms,
                    )
                    if (
                        fit.rate > t.rate_p95
                        and profile.llm_rewrite
                        and rewrites < profile.max_rewrites
                        and not ue.is_protected(u.id)
                    ):
                        cur = count_units(text, tgt_lp.speech_unit)
                        rw = await cx.call(
                            "mt.rewrite",
                            RewriteReq(
                                unit_id=u.id,
                                tgt_lang=spec.tgt_lang,
                                source_text=u.text,
                                text=text,
                                current_units=cur,
                                target_units=max(1, int(cur * t.rate_p95 / fit.rate)),
                                self_term=tr_u.self_term if tr_u else None,
                                other_term=tr_u.other_term if tr_u else None,
                            ),
                            subject=u.id,
                        )
                        rewrites += 1
                        # A shorter line must not break the forms of address: keep the old
                        # one (and stop rewriting) when the rewrite does.
                        if tr_u and address_ops.check_line(
                            u, rw.text, tr_u, sheet.cast, spec.tgt_lang, generic
                        ):
                            break
                        text, variant, pred = rw.text, "rewrite", predict(rw.text)
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
    if placements:
        tile_reqs = [
            TileReq(
                bed_key=stems.instrumental_key,
                start_ms=a,
                end_ms=b,
                placements=[p for p in placements if p.start_ms < b and p.end_ms > a],
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
        if spec.export:
            full: AudioRes = await cx.local(
                "audio.concat",
                mix_ops.concat,
                ConcatReq(keys=[x.audio_key for x in tiles]),
                AudioRes,
            )
            dub_audio_key = full.audio_key
            m4a: AudioRes = await cx.local(
                "audio.master",
                mix_ops.master,
                MasterReq(audio_key=full.audio_key, loudness_lufs=cfg.dub.mix.loudness_lufs),
                AudioRes,
            )
            dub_m4a_key = m4a.audio_key
        await checking
        tgt_items = [
            SubItem(
                start_ms=p.start_ms,
                end_ms=p.end_ms,
                text=p.text,
                words=_shift(checks[p.unit_id].words, p.start_ms) if p.unit_id in checks else [],
            )
            for p in placements
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
    _write_outputs(
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
    )
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


def cast_markdown(sheet: ContextSheet, units: list[Unit], tgt_lang: str) -> str:
    """The cast sheet for a human to check: who is who (with talk time and first lines, to
    recognise them) and how each pair addresses each other in the target language."""
    talk: dict[str, list[Unit]] = {}
    for u in units:
        talk.setdefault(u.speaker, []).append(u)
    out = ["# Cast sheet", "", "| id | name | gender | age | role | talk | first lines |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    rows = {s.id: s for s in sheet.cast.speakers}
    for spk in sorted(set(rows) | set(talk)):
        s = rows.get(spk)
        us = talk.get(spk, [])
        secs = sum(u.end_ms - u.start_ms for u in us) / 1000
        first = " / ".join(u.text for u in us[:2]).replace("|", "/")
        cells = [spk, s.name or "" if s else "", s.gender if s else "", s.age_band if s else "",
                 (s.role or "").replace("|", "/") if s else ""]  # fmt: skip
        mark = " (confirmed)" if s and s.locked else ""
        out.append(f"| {' | '.join(cells)}{mark} | {secs:.0f} s, {len(us)} lines | {first[:120]} |")
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
                 profile_name) -> JobResult:  # fmt: skip
    """Stop at the cast sheet (`dub run --confirm-cast`): everything so far is cached, so the
    full run after the corrections starts at translation."""
    tracer.save(job_dir / "trace.json")
    (job_dir / "cast.json").write_text(sheet.model_dump_json(indent=2), encoding="utf-8")
    (job_dir / "cast.md").write_text(cast_markdown(sheet, units, spec.tgt_lang), encoding="utf-8")
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


def _free_phase_a(cfg: Config) -> bool:
    v = cfg.dub.throughput.free_phase_a_models
    if v == "auto":
        mem = os.environ.get("DUB_GPU_MEM_MIB")  # set by dubd from nvidia-smi
        return bool(mem) and int(mem) < 40000
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
) -> None:
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
    (job_dir / "report.md").write_text(
        render_report(cx, seg, sheet, placements, tgt_track, tgt_lp, metrics), encoding="utf-8"
    )


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
