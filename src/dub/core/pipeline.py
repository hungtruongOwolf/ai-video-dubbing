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
from dub.ops import edits as edits_ops
from dub.ops import media as media_ops
from dub.ops import mix as mix_ops
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

UNIT_CONCURRENCY = 8


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
    job_id = job_id_for(spec, profile_name)
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
    )
    ws.db.upsert_job(
        job_id,
        source=spec.source_name,
        src_lang=spec.src_lang,
        tgt_lang=spec.tgt_lang,
        profile=profile_name,
        status="running",
    )

    # ---------------------------------------------------------------- Phase A
    log("[1/7] decoding audio")
    media: MediaInfo = await cx.local(
        "media.decode", media_ops.decode, DecodeReq(src_key=spec.audio_key), MediaInfo
    )
    log("[2/7] separating speech from background + transcribing (in parallel)")
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

    log("[3/7] finding who speaks when")
    diar: Diarization = await cx.call(
        "audio.diarize",
        DiarizeReq(
            audio_key=stems.vocals_key,
            speakers_hint=spec.speakers_hint,
            replay_key=spec.transcript_key,
        ),
    )
    asr_words = sorted(
        (w for w in words.words if w.w.strip()), key=lambda w: (w.start_ms, w.end_ms)
    )
    vocals16: AudioRes = await cx.local(
        "audio.to_mono16k", media_ops.to_mono16k, AudioReq(audio_key=stems.vocals_key), AudioRes
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
    edits = edits_ops.effective(spec.edits)
    ue = edits_ops.UnitEdits.from_edits(edits)
    worded = segment_ops.assign_speakers(asr_words, diar.turns)
    units, warnings = edits_ops.apply_structure(base_units, worded, edits, src_lp.joiner)
    for w in warnings:
        log(f"      edit skipped: {w}")
    voiced = ue.apply_to_units(units)  # speaker and slot edits
    refs: RefBank = await cx.local(
        "voice.extract_refs",
        refs_ops.extract_refs,
        RefsReq(vocals_key=stems.vocals_key, units=voiced),
        RefBank,
    )
    if edits:
        log(f"      {len(edits)} human edits applied")
    log(f"      {len(words.words)} words -> {len(seg.sentences)} sentences -> {len(units)} units")

    # ---------------------------------------------------------------- Barrier: context pass
    log("[4/7] reading the whole script: summary, glossary, cast sheet")
    sheet: ContextSheet = await cx.call(
        "mt.context",
        ContextReq(src_lang=src_lang, tgt_lang=spec.tgt_lang, units=base_units, voices=diar.voices),
    )

    # ---------------------------------------------------------------- Phase B: translate
    log("[5/7] translating")
    tcfg = cfg.dub.translate
    translations: dict[str, Translation] = {}

    # Windows are cut on the base (unedited) units, so a split or merge only changes the
    # window that holds it; neighbours come from the base units too.
    size, ctx_n = tcfg.window_units, tcfg.context_units
    base_index = {u.id: i for i, u in enumerate(base_units)}
    groups: dict[int, list[Unit]] = {}
    for u in units:
        groups.setdefault(base_index.get((u.parent_ids or [u.id])[0], 0) // size, []).append(u)

    async def translate_window(k: int, win: list[Unit]) -> None:
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
        )
        batch: TranslationBatch = await cx.call("mt.translate", req, subject=f"w{k:04d}")
        for tr in batch.items:
            translations[tr.unit_id] = tr

    await asyncio.gather(*(translate_window(k, w) for k, w in sorted(groups.items())))
    untranslated = {u.id for u in units if u.id not in translations}
    for u in units:  # never crash on a window the LLM answered partially: flag it instead
        if u.id in untranslated:
            translations[u.id] = Translation(unit_id=u.id, short=u.text, target=u.text, long=u.text)
    translations = {uid: ue.apply_to_translation(tr) for uid, tr in translations.items()}

    def predict(text: str) -> int:
        return predict_ms(text, tgt_lp.speech_unit, tgt_lp.units_per_s)

    choices = {}
    for u in voiced:
        if u.id in ue.text:
            choices[u.id] = ("edit", ue.text[u.id], predict(ue.text[u.id]))
        else:
            choices[u.id] = pick_variant(translations[u.id], u.slot_ms, predict, t.variant_window)
    units = voiced

    placements: list[Placement] = []
    checks: dict[str, WordTimeline] = {}  # unit id -> back-check words (relative to the clip)
    trims: dict[str, TrimRes] = {}
    engine = media_ops.stretch_engine()
    if not spec.subs_only:
        log(f"[6/7] voicing and fitting {len(units)} units")
        sem = asyncio.Semaphore(UNIT_CONCURRENCY)

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
            facts: dict[str, Any] = {
                "unit.slot_ms": u.slot_ms,
                "unit.words": u.n_words,
                "unit.asr_conf": u.asr_conf,
            }
            rewrites = 0
            ref = refs.for_speaker(u.speaker)
            async with sem:
                while True:

                    def make_req(m: ProviderManifest, text=text, pred=pred) -> SynthReq:
                        exact = m.caps.duration_control == "exact"
                        return SynthReq(
                            unit_id=u.id,
                            text=text,
                            lang=spec.tgt_lang,
                            speaker=u.speaker,
                            seed=ue.seed.get(u.id, 0),
                            voice_key=ref.audio_key if ref else None,
                            voice_text=ref.text if ref else None,
                            target_ms=exact_target_ms(u.slot_ms, pred, t.safe_ratio)
                            if exact
                            else None,
                        )

                    synth: SynthRes = await cx.call(
                        "tts.synthesize", make_req, facts=facts, subject=u.id
                    )
                    # Fit the voice, not the clip: silence the engine left around it would
                    # shift the onset and waste the slot.
                    trimmed: TrimRes = await cx.local(
                        "audio.trim",
                        media_ops.trim,
                        TrimReq(audio_key=synth.audio_key),
                        TrimRes,
                        subject=u.id,
                    )
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
                            ),
                            subject=u.id,
                        )
                        text, variant, pred = rw.text, "rewrite", predict(rw.text)
                        rewrites += 1
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
            flags = list(fit.flags) + (["untranslated"] if u.id in untranslated else [])
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

        placements = list(await asyncio.gather(*(dub_unit(i) for i in range(len(units)))))

    async def backcheck(p: Placement) -> None:
        """Transcribe one dubbed clip again (target language): subtitle timing + CER."""
        clip16: AudioRes = await cx.local(
            "audio.to_mono16k",
            media_ops.to_mono16k,
            AudioReq(audio_key=p.audio_key),
            AudioRes,
            subject=p.unit_id,
        )
        checks[p.unit_id] = await cx.call(
            "asr.backcheck",
            TranscribeReq(audio_key=clip16.audio_key, lang=spec.tgt_lang, expected_text=p.text),
            subject=p.unit_id,
        )

    # Back-checks run while the tiles are mixed: they only feed subtitles and the report.
    checking = asyncio.gather(*(backcheck(p) for p in placements if profile.backcheck))

    # ---------------------------------------------------------------- mix + subtitles
    log("[7/7] mixing and writing subtitles")
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
        risk, why = risk_ops.score(u, p, tr, edited=edited, cer=backcheck_cer.get(u.id))
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
