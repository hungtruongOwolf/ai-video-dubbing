"""`dub` command line.

dub demo                         build a 1-minute test video in ./demo
dub run VIDEO --to vi --fake     dub a video with fake models (no GPU, no cost)
dub run VIDEO --to vi --fake --use "mt.*=claude_cli"
                                 same, but translate for real with your Claude plan
dub up / dub down / dub status   rent, terminate and inspect the GPU pod
dub run VIDEO --to vi --remote   dub on the GPU pod
dub edit VIDEO --to vi ACTION    record a fix for one line (text, regen, speaker, nudge, ...)
dub review VIDEO --to vi         review page: listen to risky lines, fix them, re-render
dub doctor                       show which providers can run here and the route matrix
dub schema                       export the /v1 data contracts as JSON Schema
dub jobs                         list jobs in the workspace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dub import __version__
from dub.contracts.v1 import ALL_MODELS, SCHEMA_VERSION
from dub.core.config import default_config_dir, load_config
from dub.core.ctx import OP_RESULT
from dub.core.router import Router, RoutingError, unavailable_reason
from dub.edits import EditStore
from dub.runner import RunOptions, audio_key_for, finish, run_video
from dub.store import Workspace


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config-dir", type=Path, default=None, help="default: <repo>/configs")
    p.add_argument(
        "--workspace",
        type=Path,
        default=Path("dub-workspace"),
        help="cache + job folder (default: ./dub-workspace)",
    )


def _video_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("video", type=Path)
    p.add_argument("--to", dest="tgt", required=True, help="target language, e.g. vi, en, ja")
    p.add_argument("--from", dest="src", default="auto", help="source language (default: auto)")
    p.add_argument("--profile", default=None, help="draft | final (default from dub.yaml)")
    p.add_argument(
        "--fake", action="store_true", help="use fake providers for every op not set with --use"
    )
    p.add_argument(
        "--use",
        action="append",
        default=[],
        metavar="OP=PROVIDER",
        help='force a provider, e.g. --use "mt.*=claude_cli" (repeatable)',
    )
    p.add_argument(
        "--remote",
        action="store_true",
        help="run on the GPU pod (starts one with `dub up` if needed)",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dub", description="AI video dubbing (learning project)")
    ap.add_argument("--version", action="version", version=f"dub {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="dub one video file")
    _video_args(r)
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

    e = sub.add_parser(
        "edit",
        help="record a human edit for one line (see `dub edit -h`)",
        epilog=EDIT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    e.add_argument("video", type=Path)
    e.add_argument("--to", dest="tgt", required=True)
    e.add_argument("action")
    e.add_argument("args", nargs="*")
    _common(e)

    rv = sub.add_parser("review", help="open the review page: listen, fix lines, re-render")
    _video_args(rv)
    rv.add_argument("--port", type=int, default=8765)
    rv.add_argument("--no-browser", action="store_true")
    rv.add_argument("--out-dir", type=Path, default=None)
    _common(rv)

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


def _opts(a, **over) -> RunOptions:
    return RunOptions(
        video=a.video, tgt=a.tgt, src=a.src, profile=a.profile, remote=a.remote, fake=a.fake,
        use=a.use, transcript=getattr(a, "transcript", None),
        speakers=getattr(a, "speakers", None), subs_only=getattr(a, "subs_only", False),
        config_dir=a.config_dir, **over,
    )  # fmt: skip


def cmd_run(a) -> int:
    if not a.video.exists():
        print(f"error: {a.video} does not exist", file=sys.stderr)
        return 2
    ws = Workspace.open(a.workspace.resolve())
    opts = _opts(a)
    try:
        out = run_video(opts, ws)
    except RoutingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    finish(opts, out, a.out_dir)
    return 0


EDIT_HELP = """actions:
  list                      show the edits in effect
  text UNIT "new line"      replace the dubbed line (and lock it)
  regen UNIT                a new take of the same line (next seed)
  speaker UNIT S2           who says the line (picks the cloned voice)
  nudge UNIT START END      move the slot: ms offsets, e.g. -150 200
  split UNIT [AT_MS]        split at the word boundary nearest AT_MS (default: middle)
  merge UNIT                merge with the next unit (same speaker)
  lock UNIT / unlock UNIT   protect the line from automatic rewrites
  undo [EDIT_ID]            cancel the latest edit (or the given one)
Then run `dub run ...` (or the review page) again: only what changed is redone."""


def cmd_edit(a) -> int:
    ws = Workspace.open(a.workspace.resolve())
    store = EditStore.for_job(ws.root, audio_key_for(ws, a.video.resolve()), a.tgt)
    act, args = a.action, a.args

    def need(n: int) -> None:
        if len(args) < n:
            raise SystemExit(f"error: '{act}' needs {n} argument(s)\n\n{EDIT_HELP}")

    if act == "list":
        for e in store.effective():
            print(
                f"{e.id}  {e.unit_id:<28} {e.field:<10} {json.dumps(e.value, ensure_ascii=False)}"
            )
        return 0
    if act == "undo":
        e = store.undo(args[0] if args else None)
        print("nothing to undo" if e is None else f"undone: {e.value}")
        return 0
    need(1)
    unit = args[0]
    if act == "text":
        need(2)
        e = store.add(unit, "text", " ".join(args[1:]))
    elif act == "regen":
        seeds = [int(x.value) for x in store.effective() if x.unit_id == unit and x.field == "seed"]
        e = store.add(unit, "seed", (seeds[-1] if seeds else 0) + 1)
    elif act == "speaker":
        need(2)
        e = store.add(unit, "speaker", args[1])
    elif act == "nudge":
        need(3)
        e = store.add(unit, "nudge", {"start_ms": int(args[1]), "end_ms": int(args[2])})
    elif act == "split":
        e = store.add(unit, "split", {"at_ms": int(args[1])} if len(args) > 1 else None)
    elif act == "merge":
        e = store.add(unit, "merge_next")
    elif act in ("lock", "unlock"):
        e = store.add(unit, "lock", act == "lock")
    else:
        raise SystemExit(f"error: unknown action {act!r}\n\n{EDIT_HELP}")
    print(f"saved {e.field} for {e.unit_id} ({e.id}). Run `dub run` or `dub review` to render.")
    return 0


def cmd_review(a) -> int:
    from dub.review.server import serve_review

    if not a.video.exists():
        print(f"error: {a.video} does not exist", file=sys.stderr)
        return 2
    ws = Workspace.open(a.workspace.resolve())
    return serve_review(_opts(a, export=False), ws, port=a.port, open_browser=not a.no_browser,
                        out_dir=a.out_dir)  # fmt: skip


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
        "edit": cmd_edit,
        "review": cmd_review,
        "demo": cmd_demo,
        "doctor": cmd_doctor,
        "schema": cmd_schema,
        "jobs": cmd_jobs,
    }[a.cmd]
    return handler(a)


if __name__ == "__main__":
    raise SystemExit(main())
