"""`dub` command line.

dub demo                         build a 1-minute test video in ./demo
dub run VIDEO --to vi --fake     dub a video with fake models (no GPU, no cost)
dub run VIDEO --to vi --fake --use "mt.*=claude_cli"
                                 same, but translate for real with your Claude plan
dub doctor                       show which providers can run here and the route matrix
dub schema                       export the /v1 data contracts as JSON Schema
dub jobs                         list jobs in the workspace
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dub import __version__
from dub.contracts.v1 import ALL_MODELS, SCHEMA_VERSION, Word
from dub.core.config import default_config_dir, load_config
from dub.core.ctx import OP_RESULT
from dub.core.pipeline import JobSpec, run_job
from dub.core.router import Router, RoutingError, unavailable_reason
from dub.pod.server import overrides_for
from dub.store import Workspace


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config-dir", type=Path, default=None, help="default: <repo>/configs")
    p.add_argument(
        "--workspace",
        type=Path,
        default=Path("dub-workspace"),
        help="cache + job folder (default: ./dub-workspace)",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dub", description="AI video dubbing (learning project)")
    ap.add_argument("--version", action="version", version=f"dub {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="dub one video file")
    r.add_argument("video", type=Path)
    r.add_argument("--to", dest="tgt", required=True, help="target language, e.g. vi, en, ja")
    r.add_argument("--from", dest="src", default="auto", help="source language (default: auto)")
    r.add_argument("--profile", default=None, help="draft | final (default from dub.yaml)")
    r.add_argument(
        "--fake", action="store_true", help="use fake providers for every op not set with --use"
    )
    r.add_argument(
        "--use",
        action="append",
        default=[],
        metavar="OP=PROVIDER",
        help='force a provider, e.g. --use "mt.*=claude_cli" (repeatable)',
    )
    r.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="words JSON to replay (default: <video>.words.json if present)",
    )
    r.add_argument("--speakers", type=int, default=None, help="number of speakers, if known")
    r.add_argument("--subs-only", action="store_true", help="subtitles only, no voice")
    r.add_argument("--out-dir", type=Path, default=None, help="default: <video folder>/<name>.dub")
    r.add_argument(
        "--remote",
        action="store_true",
        help="run on the GPU pod (starts one with `dub up` if needed)",
    )
    _common(r)

    up = sub.add_parser("up", help="rent the GPU pod and wait until the models are loaded")
    up.add_argument("--gpu", default=None, help='GPU type id, e.g. "NVIDIA A40"')
    _common(up)
    dn = sub.add_parser("down", help="terminate the GPU pod (the volume and cache stay)")
    _common(dn)
    st = sub.add_parser("status", help="show the GPU pod and its workers")
    _common(st)

    d = sub.add_parser("demo", help="build a test video + transcript")
    d.add_argument("--out", type=Path, default=Path("demo"))

    doc = sub.add_parser("doctor", help="check providers and print the route matrix")
    doc.add_argument("--profile", default=None)
    _common(doc)

    s = sub.add_parser("schema", help="export JSON Schemas of the data contracts")
    s.add_argument("--out", type=Path, default=Path("schemas"))

    j = sub.add_parser("jobs", help="list jobs in the workspace")
    _common(j)
    return ap


# -------------------------------------------------------------------- commands


def cmd_demo(a) -> int:
    from dub.demo import make_demo

    video, words = make_demo(a.out)
    print(f"video:      {video}")
    print(f"transcript: {words}")
    print(f'next:       dub run "{video}" --to vi --fake')
    return 0


@dataclass
class RunOutput:
    """Where the results of a run are, local or downloaded from the pod."""

    job_dir: Path
    src_lang: str
    tgt_lang: str
    dub_audio: Path | None
    tgt_srt: Path
    src_srt: Path
    calls: int
    cache_hits: int


def _run_local(a, ws: Workspace, spec: JobSpec) -> RunOutput:
    cfg = load_config(a.config_dir)
    overrides = overrides_for(cfg, a.use, a.fake)
    res = asyncio.run(run_job(cfg, ws, spec, overrides=overrides, log=print))
    audio = res.dub_m4a_key or res.dub_audio_key
    return RunOutput(
        job_dir=res.job_dir,
        src_lang=res.src_lang,
        tgt_lang=res.tgt_lang,
        dub_audio=ws.cas.path(audio) if audio else None,
        tgt_srt=ws.cas.path(res.subs[res.tgt_lang]),
        src_srt=ws.cas.path(res.subs[f"{res.src_lang}.src"]),
        calls=res.calls,
        cache_hits=res.cache_hits,
    )


def _run_remote(a, ws: Workspace, spec: JobSpec, audio: Path, transcript: Path | None):
    from dub import remote

    state = remote.PodState.load(ws.root)
    if state is None or not remote.Dubd(state.url, state.token).health():
        print("[laptop] no running pod: starting one (dub up)")
        cfg_dir = a.config_dir or default_config_dir()
        state = remote.pod_up(remote.PodConfig.load(cfg_dir), ws.root, cfg_dir.parent)
    dubd = remote.Dubd(state.url, state.token)
    body: dict = {
        "audio_key": dubd.upload(audio, audio.suffix),
        "source_name": spec.source_name,
        "tgt_lang": spec.tgt_lang,
        "src_lang": spec.src_lang,
        "profile": spec.profile,
        "speakers_hint": spec.speakers_hint,
        "subs_only": spec.subs_only,
        "use": a.use,
        "fake": a.fake,
    }
    if transcript:
        body["transcript_key"] = dubd.upload(transcript, ".json", log=lambda m: None)
    job_id = dubd.submit(body)
    print(f"[laptop] job {job_id} submitted to pod {state.pod_id}")
    seen = 0
    while True:
        job = dubd.job(job_id)
        for line in job["log"][seen:]:
            print(f"[pod] {line}")
        seen = len(job["log"])
        if job["status"] in ("done", "error"):
            break
        time.sleep(3)
    if job["status"] == "error":
        raise RuntimeError(f"job failed on the pod: {job['error']}")
    r = job["result"]
    job_dir = ws.job_dir(r["job_dir"])
    for name in ("report.md", "trace.json", "units.json", "cast.json", "routes.jsonl"):
        dubd.download(f"/jobs/{job_id}/files/{name}", job_dir / name)
    tgt_srt = dubd.download(f"/cas/{r['subs'][r['tgt_lang']]}", job_dir / "tgt.srt")
    src_srt = dubd.download(f"/cas/{r['subs'][r['src_lang'] + '.src']}", job_dir / "src.srt")
    dub_audio = None
    if r.get("dub_m4a_key"):
        print("[laptop] downloading the dubbed audio")
        dub_audio = dubd.download(f"/cas/{r['dub_m4a_key']}", job_dir / "dub.m4a")
    return RunOutput(
        job_dir,
        r["src_lang"],
        r["tgt_lang"],
        dub_audio,
        tgt_srt,
        src_srt,
        r["calls"],
        r["cache_hits"],
    )


def cmd_run(a) -> int:
    from dub import laptop

    video = a.video.resolve()
    if not video.exists():
        print(f"error: {video} does not exist", file=sys.stderr)
        return 2
    ws = Workspace.open(a.workspace.resolve())

    print(f"[laptop] extracting the audio track of {video.name} (no re-encode)")
    tmp = ws.cas.tmp_path(".mka")
    laptop.extract_audio(video, tmp)
    audio_key = ws.cas.put_file(tmp, move=True)

    transcript = a.transcript or video.with_name(f"{video.stem}.words.json")
    transcript_key = None
    if transcript.exists():
        data = json.loads(transcript.read_text(encoding="utf-8"))
        [Word.model_validate(w) for w in data["words"]]  # fail early on a bad file
        transcript_key = ws.cas.put_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True), ".json"
        )
        print(f"[laptop] transcript found: {transcript.name} (the fake ASR replays it)")

    spec = JobSpec(
        audio_key=audio_key,
        source_name=video.name,
        tgt_lang=a.tgt,
        src_lang=a.src,
        profile=a.profile,
        transcript_key=transcript_key,
        speakers_hint=a.speakers,
        subs_only=a.subs_only,
    )
    try:
        if a.remote:
            t = ws.cas.path(transcript_key) if transcript_key else None
            out = _run_remote(a, ws, spec, ws.cas.path(audio_key), t)
        else:
            out = _run_local(a, ws, spec)
    except RoutingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    out_dir = (a.out_dir or video.parent / f"{video.stem}.dub").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_srt = out_dir / f"{video.stem}.{out.tgt_lang}.srt"
    src_srt = out_dir / f"{video.stem}.{out.src_lang}.srt"
    shutil.copyfile(out.tgt_srt, tgt_srt)
    shutil.copyfile(out.src_srt, src_srt)
    if out.dub_audio:
        print("[laptop] muxing into the original video (video stream copied, not re-encoded)")
        out_video = out_dir / f"{video.stem}.{out.tgt_lang}{laptop.output_container(video)}"
        laptop.mux(
            video,
            out.dub_audio,
            out_video,
            tgt_lang=out.tgt_lang,
            src_lang=out.src_lang,
            subs=[(tgt_srt, out.tgt_lang), (src_srt, out.src_lang)],
        )
        print(f"\nvideo:     {out_video}")
    print(f"subtitles: {tgt_srt}\n           {src_srt}")
    print(f"report:    {out.job_dir / 'report.md'}")
    print(f"trace:     {out.job_dir / 'trace.json'}  (open at https://ui.perfetto.dev)")
    print(f"calls:     {out.calls}, cache hits: {out.cache_hits}")
    return 0


def cmd_up(a) -> int:
    from dub import remote

    cfg_dir = a.config_dir or default_config_dir()
    try:
        remote.pod_up(
            remote.PodConfig.load(cfg_dir),
            a.workspace.resolve(),
            cfg_dir.parent,
            gpus=[a.gpu] if a.gpu else None,
        )
    except remote.RemoteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    return 0


def cmd_down(a) -> int:
    from dub import remote

    remote.pod_down(a.workspace.resolve())
    return 0


def cmd_status(a) -> int:
    from dub import remote

    remote.pod_status(a.workspace.resolve())
    return 0


def cmd_doctor(a) -> int:
    cfg = load_config(a.config_dir)
    print(f"config: {cfg.root}\n\nProviders:")
    for pid, m in sorted(cfg.providers.items()):
        why = unavailable_reason(m)
        state = "ok" if why is None else f"unavailable: {why}"
        tag = " (fake)" if m.fake else ""
        print(f"  {pid:<20} {','.join(m.op):<45} {state}{tag}")
    router = Router(cfg, profile=a.profile)
    langs = sorted(lp for lp in cfg.langpacks if lp != "default")
    print(f"\nRoute matrix (profile {router.profile_name}; first provider of the chain):")
    print("  " + "op".ljust(18) + "".join(lang.ljust(16) for lang in langs))
    for op in OP_RESULT:
        cells = []
        for lang in langs:
            try:
                tr = router.route(op, {"src_lang": lang, "tgt_lang": lang})
                cells.append(f"{tr.chain[0]}")
            except RoutingError:
                cells.append("-")
        print("  " + op.ljust(18) + "".join(c[:15].ljust(16) for c in cells))
    print("\n'-' = nothing installed yet for that op (real models arrive in M1). Use --fake.")
    return 0


def cmd_schema(a) -> int:
    out = a.out / SCHEMA_VERSION
    out.mkdir(parents=True, exist_ok=True)
    for model in ALL_MODELS:
        (out / f"{model.__name__}.json").write_text(
            json.dumps(model.model_json_schema(), indent=2), encoding="utf-8"
        )
    print(f"wrote {len(ALL_MODELS)} schemas to {out}")
    return 0


def cmd_jobs(a) -> int:
    ws = Workspace.open(a.workspace)
    for job_id, tgt, profile, status, _ in ws.db.jobs():
        print(f"{job_id:<60} {tgt:<4} {profile:<6} {status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    handler = {
        "up": cmd_up,
        "down": cmd_down,
        "status": cmd_status,
        "run": cmd_run,
        "demo": cmd_demo,
        "doctor": cmd_doctor,
        "schema": cmd_schema,
        "jobs": cmd_jobs,
    }[a.cmd]
    return handler(a)


if __name__ == "__main__":
    raise SystemExit(main())
