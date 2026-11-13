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
from fnmatch import fnmatch
from pathlib import Path

from dub import __version__
from dub.contracts.v1 import ALL_MODELS, SCHEMA_VERSION, Word
from dub.core.config import load_config
from dub.core.ctx import OP_RESULT
from dub.core.pipeline import JobSpec, run_job
from dub.core.router import Router, RoutingError, parse_overrides, unavailable_reason
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
    _common(r)

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


def cmd_run(a) -> int:
    from dub import laptop

    video = a.video.resolve()
    if not video.exists():
        print(f"error: {video} does not exist", file=sys.stderr)
        return 2
    cfg = load_config(a.config_dir)
    ws = Workspace.open(a.workspace)

    overrides = parse_overrides(a.use)
    if a.fake:
        for op in OP_RESULT:
            if not any(fnmatch(op, pat) for pat, _ in overrides):
                fp = cfg.fake_provider_for(op)
                if fp:
                    overrides.append((op, [fp]))

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
        res = asyncio.run(run_job(cfg, ws, spec, overrides=overrides, log=print))
    except RoutingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    out_dir = (a.out_dir or video.parent / f"{video.stem}.dub").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_srt = out_dir / f"{video.stem}.{res.tgt_lang}.srt"
    src_srt = out_dir / f"{video.stem}.{res.src_lang}.srt"
    shutil.copyfile(ws.cas.path(res.subs[res.tgt_lang]), tgt_srt)
    shutil.copyfile(ws.cas.path(res.subs[f"{res.src_lang}.src"]), src_srt)
    print("[laptop] muxing into the original video (video stream copied, not re-encoded)")
    if res.dub_audio_key:
        out_video = out_dir / f"{video.stem}.{res.tgt_lang}{laptop.output_container(video)}"
        laptop.mux(
            video,
            ws.cas.path(res.dub_audio_key),
            out_video,
            tgt_lang=res.tgt_lang,
            src_lang=res.src_lang,
            subs=[(tgt_srt, res.tgt_lang), (src_srt, res.src_lang)],
        )
        print(f"\nvideo:     {out_video}")
    print(f"subtitles: {tgt_srt}\n           {src_srt}")
    print(f"report:    {res.job_dir / 'report.md'}")
    print(f"trace:     {res.job_dir / 'trace.json'}  (open at https://ui.perfetto.dev)")
    print(f"calls:     {res.calls}, cache hits: {res.cache_hits}")
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
        "run": cmd_run,
        "demo": cmd_demo,
        "doctor": cmd_doctor,
        "schema": cmd_schema,
        "jobs": cmd_jobs,
    }[a.cmd]
    return handler(a)


if __name__ == "__main__":
    raise SystemExit(main())
