"""Run the pipeline for one video, locally or on the GPU pod, and return where the results are.

Shared by `dub run` and the review page. Both always send the current human edits.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dub import laptop
from dub.contracts.v1 import Word
from dub.core.config import default_config_dir, load_config
from dub.core.pipeline import JobSpec, run_job
from dub.edits import EditStore
from dub.pod.server import overrides_for
from dub.store import Workspace


@dataclass
class RunOptions:
    video: Path
    tgt: str
    src: str = "auto"
    profile: str | None = None
    remote: bool = False
    fake: bool = False
    use: list[str] = field(default_factory=list)
    transcript: Path | None = None
    speakers: int | None = None
    subs_only: bool = False
    export: bool = True
    config_dir: Path | None = None


@dataclass
class RunOutput:
    """Where the results of a run are, local or downloaded from the pod."""

    job_id: str
    job_dir: Path
    src_lang: str
    tgt_lang: str
    dub_audio: Path | None  # AAC track, only when exported
    tgt_srt: Path
    src_srt: Path
    tiles: list[dict]  # mixed 30 s tiles: start_ms, end_ms, audio_key (local or pod CAS)
    remote: bool
    calls: int
    cache_hits: int
    warnings: list[str] = field(default_factory=list)
    redone: dict[str, int] = field(default_factory=dict)


def audio_key_for(ws: Workspace, video: Path, log: Callable[[str], None] = print) -> str:
    """The CAS key of the video's audio track, extracting it only when the file changed."""
    index_path = ws.root / "sources.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        index = {}
    st = video.stat()
    entry = index.get(str(video))
    if entry and entry["size"] == st.st_size and entry["mtime_ns"] == st.st_mtime_ns \
            and entry.get("v") == laptop.EXTRACT_VERSION \
            and ws.cas.exists(entry["audio_key"]):  # fmt: skip
        return entry["audio_key"]
    log(f"[laptop] extracting the audio track of {video.name} (no re-encode)")
    placeholder = ws.cas.tmp_path(".audio")  # extract_audio picks the real extension
    path = laptop.extract_audio(video, placeholder)
    placeholder.unlink(missing_ok=True)
    key = ws.cas.put_file(path, move=True)
    index[str(video)] = {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "audio_key": key,
        "v": laptop.EXTRACT_VERSION,
    }
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return key


def transcript_key_for(ws: Workspace, opts: RunOptions, log: Callable[[str], None]) -> str | None:
    path = opts.transcript or opts.video.with_name(f"{opts.video.stem}.words.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    [Word.model_validate(w) for w in data["words"]]  # fail early on a bad file
    log(f"[laptop] transcript found: {path.name} (the fake ASR replays it)")
    return ws.cas.put_text(json.dumps(data, ensure_ascii=False, sort_keys=True), ".json")


def run_video(opts: RunOptions, ws: Workspace, log: Callable[[str], None] = print) -> RunOutput:
    video = opts.video.resolve()
    audio_key = audio_key_for(ws, video, log)
    edits = EditStore.for_job(ws.root, audio_key, opts.tgt).all()
    spec = JobSpec(
        audio_key=audio_key, source_name=video.name, tgt_lang=opts.tgt, src_lang=opts.src,
        profile=opts.profile, transcript_key=transcript_key_for(ws, opts, log),
        speakers_hint=opts.speakers, subs_only=opts.subs_only, edits=edits, export=opts.export,
    )  # fmt: skip
    if opts.remote:
        return _run_remote(opts, ws, spec, log)
    cfg = load_config(opts.config_dir)
    res = asyncio.run(run_job(cfg, ws, spec, overrides=overrides_for(cfg, opts.use, opts.fake),
                              log=log))  # fmt: skip
    return RunOutput(
        job_id=res.job_id, job_dir=res.job_dir, src_lang=res.src_lang, tgt_lang=res.tgt_lang,
        dub_audio=ws.cas.path(res.dub_m4a_key) if res.dub_m4a_key else None,
        tgt_srt=ws.cas.path(res.subs[res.tgt_lang]),
        src_srt=ws.cas.path(res.subs[f"{res.src_lang}.src"]),
        tiles=[t.model_dump() for t in res.tiles], remote=False,
        calls=res.calls, cache_hits=res.cache_hits, warnings=res.warnings,
        redone=res.redone,
    )  # fmt: skip


def _run_remote(opts: RunOptions, ws: Workspace, spec: JobSpec, log) -> RunOutput:
    from dub import remote

    state = remote.PodState.load(ws.root)
    if state is None or not remote.Dubd(state.url, state.token).health():
        log("[laptop] no running pod: starting one (dub up)")
        cfg_dir = opts.config_dir or default_config_dir()
        state = remote.pod_up(remote.PodConfig.load(cfg_dir), ws.root, cfg_dir.parent, log=log)
    dubd = remote.Dubd(state.url, state.token)

    def ensure(key: str) -> str:
        """Upload a CAS file to the pod unless the pod already has it."""
        if dubd.has(key):
            return key
        got = dubd.upload(ws.cas.path(key), Path(key).suffix, log=log)
        if got != key:
            raise remote.RemoteError(f"upload hash mismatch: {got} != {key}")
        return key

    body: dict = {
        "audio_key": ensure(spec.audio_key), "source_name": spec.source_name,
        "tgt_lang": spec.tgt_lang, "src_lang": spec.src_lang, "profile": spec.profile,
        "speakers_hint": spec.speakers_hint, "subs_only": spec.subs_only,
        "use": opts.use, "fake": opts.fake, "export": spec.export,
        "edits": [e.model_dump(mode="json") for e in spec.edits],
    }  # fmt: skip
    if spec.transcript_key:
        body["transcript_key"] = ensure(spec.transcript_key)
    job_id = dubd.submit(body)
    log(f"[laptop] job {job_id} submitted to pod {state.pod_id}")
    seen = 0
    while True:
        job = dubd.job(job_id, wait=20, seen=seen, bundle=True)
        for line in job["log"][seen:]:
            log(f"[pod] {line}")
        seen = len(job["log"])
        if job["status"] in ("done", "error"):
            break
    if job["status"] == "error":
        raise RuntimeError(f"job failed on the pod: {job['error']}")
    r = job["result"]
    job_dir = ws.job_dir(r["job_dir"])
    files = job.get("files") or dubd.bundle(job_id)
    for name, text in files.items():
        if not name.startswith("subs:"):
            (job_dir / name).write_text(text, encoding="utf-8")
    tgt_srt = job_dir / "tgt.srt"
    src_srt = job_dir / "src.srt"
    tgt_srt.write_text(files[f"subs:{r['tgt_lang']}"], encoding="utf-8")
    src_srt.write_text(files[f"subs:{r['src_lang']}.src"], encoding="utf-8")
    if "seconds" in r:
        log(f"[pod] job took {r['seconds']} s on the pod")
    dub_audio = None
    if r.get("dub_m4a_key"):
        log("[laptop] downloading the dubbed audio")
        dub_audio = dubd.download(f"/cas/{r['dub_m4a_key']}", job_dir / "dub.m4a")
    return RunOutput(
        job_id=r["job_dir"], job_dir=job_dir, src_lang=r["src_lang"], tgt_lang=r["tgt_lang"],
        dub_audio=dub_audio, tgt_srt=tgt_srt, src_srt=src_srt, tiles=r.get("tiles", []),
        remote=True, calls=r["calls"], cache_hits=r["cache_hits"],
        warnings=r.get("warnings", []),
        redone=r.get("redone", {}),
    )  # fmt: skip


def finish(opts: RunOptions, out: RunOutput, out_dir: Path | None,
           log: Callable[[str], None] = print) -> Path | None:  # fmt: skip
    """Copy the subtitles next to the video and mux the dub into it (when exported)."""
    import shutil

    video = opts.video.resolve()
    out_dir = (out_dir or video.parent / f"{video.stem}.dub").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_srt = out_dir / f"{video.stem}.{out.tgt_lang}.srt"
    src_srt = out_dir / f"{video.stem}.{out.src_lang}.srt"
    shutil.copyfile(out.tgt_srt, tgt_srt)
    shutil.copyfile(out.src_srt, src_srt)
    out_video = None
    if out.dub_audio:
        log("[laptop] muxing into the original video (video stream copied, not re-encoded)")
        out_video = out_dir / f"{video.stem}.{out.tgt_lang}{laptop.output_container(video)}"
        laptop.mux(video, out.dub_audio, out_video, tgt_lang=out.tgt_lang, src_lang=out.src_lang,
                   subs=[(tgt_srt, out.tgt_lang), (src_srt, out.src_lang)])  # fmt: skip
        log(f"\nvideo:     {out_video}")
    log(f"subtitles: {tgt_srt}\n           {src_srt}")
    log(f"report:    {out.job_dir / 'report.md'}")
    log(f"trace:     {out.job_dir / 'trace.json'}  (open at https://ui.perfetto.dev)")
    log(f"calls:     {out.calls}, cache hits: {out.cache_hits}")
    if out.redone:
        log("redone:    " + ", ".join(f"{op} {n}" for op, n in sorted(out.redone.items())))
    return out_video
