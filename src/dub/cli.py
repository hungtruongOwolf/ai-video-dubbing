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
dub gc [--remote]                delete cache files no result points to (frees disk)
dub bench seg VIDEO --golden FILE --remote
                                 analyse a video (no dubbing) and score its sentence
                                 boundaries against the sentences really spoken
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
    p.add_argument(
        "--to", dest="tgt", required=True,
        help="target language, e.g. vi; several for run: vi,es,ja (the analysis runs once)",
    )  # fmt: skip
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
        "--param",
        action="append",
        default=[],
        metavar="PROVIDER.KEY=VALUE",
        help="override a provider parameter for this run, e.g. whisper_large_v3.batch_size=16; "
        "dub.SECTION.KEY=VALUE overrides a dub.yaml setting, e.g. dub.voice.audition=1",
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
    r.add_argument(
        "--confirm-cast",
        action="store_true",
        help="stop after the cast sheet (who is who, forms of address) so you can correct it",
    )
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

    g = sub.add_parser("gc", help="delete cache files that no stored result points to")
    g.add_argument("--remote", action="store_true", help="on the GPU pod's network volume")
    g.add_argument("--min-age-min", type=float, default=60, help="keep newer files (default 60)")
    g.add_argument("--dry-run", action="store_true", help="only count")
    g.add_argument(
        "--older-than-hours", type=float, default=None,
        help="also forget audio results stored more than N hours ago (transcripts and "
        "translations stay), freeing their files",
    )  # fmt: skip
    _common(g)

    b = sub.add_parser("bench", help="measure a job: segmentation F1, or where the time goes")
    b.add_argument(
        "what", choices=["seg", "speed"],
        help="seg: sentence segmentation F1; speed: stages, ops, lanes and GPU use",
    )  # fmt: skip
    b.add_argument("target", type=Path, help="a video/audio file, or a job folder to re-score")
    b.add_argument("--golden", type=Path, help="seg: reference, one sentence per line")
    b.add_argument("--to", dest="tgt", default="vi", help="speed: target language (default vi)")
    b.add_argument("--profile", default=None, help="speed: draft | final")
    b.add_argument(
        "--fresh", action="store_true", help="speed: recompute every step (ignore the cache)"
    )
    b.add_argument("--tol", type=int, default=1, help="tolerance in words (default 1)")
    b.add_argument("--from", dest="src", default="auto", help="spoken language")
    b.add_argument("--remote", action="store_true", help="analyse on the GPU pod")
    b.add_argument("--fake", action="store_true", help="fake models (tests only)")
    b.add_argument("--json", type=Path, help="also write the scores to this file")
    b.add_argument(
        "--use",
        action="append",
        default=[],
        help="override routing, e.g. asr.transcribe=whisper_large_v3",
    )
    b.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="PROVIDER.KEY=VALUE",
        help="override a provider parameter for this run, e.g. whisper_large_v3.batch_size=16; "
        "dub.SECTION.KEY=VALUE overrides a dub.yaml setting, e.g. dub.voice.audition=1",
    )
    _common(b)
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
        use=a.use, params=getattr(a, "param", []), transcript=getattr(a, "transcript", None),
        speakers=getattr(a, "speakers", None), subs_only=getattr(a, "subs_only", False),
        confirm_cast=getattr(a, "confirm_cast", False), config_dir=a.config_dir, **over,
    )  # fmt: skip


def cmd_run_many(a, targets: list[str]) -> int:
    """One source, several target languages: the jobs run at the same time and share the
    analysis (phase A runs once: the pod computes each identical call once, single flight)."""
    import copy
    import threading

    ws = Workspace.open(a.workspace.resolve())
    audio_key_for(ws, a.video.resolve())  # extract once, before the jobs start
    results: dict[str, int] = {}
    lock = threading.Lock()

    def one(lang: str) -> None:
        b = copy.copy(a)
        b.tgt = lang

        def log(msg: str) -> None:
            with lock:
                print(f"[{lang}] {msg}", flush=True)

        try:
            opts = _opts(b)
            out = run_video(opts, Workspace.open(a.workspace.resolve()), log=log)
            finish(opts, out, a.out_dir, log=log)
            results[lang] = 0
        except Exception as e:  # noqa: BLE001 - report per language, keep the others
            log(f"error: {e}")
            results[lang] = 3

    threads = [threading.Thread(target=one, args=(lang,)) for lang in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return max(results.values(), default=0)


def cmd_run(a) -> int:
    if not a.video.exists():
        print(f"error: {a.video} does not exist", file=sys.stderr)
        return 2
    targets = [t.strip() for t in a.tgt.split(",") if t.strip()]
    if len(targets) > 1:
        if a.confirm_cast:
            print("error: --confirm-cast takes one target language", file=sys.stderr)
            return 2
        return cmd_run_many(a, targets)
    ws = Workspace.open(a.workspace.resolve())
    opts = _opts(a)
    try:
        out = run_video(opts, ws)
    except RoutingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    if a.confirm_cast:
        print("\n" + (out.job_dir / "cast.md").read_text(encoding="utf-8"))
        v, to = a.video, a.tgt
        print(
            "Correct it, then run again without --confirm-cast (the analysis is cached):\n"
            f"  dub edit {v} --to {to} cast S2 gender=female name=Linda\n"
            f"  dub edit {v} --to {to} cast S1>S2 self_term=mẹ other_term=con\n"
            f"  dub run {v} --to {to}"
        )
        return 0
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
Then run `dub run ...` (or the review page) again: only what changed is redone.
  cast S2 gender=female name=Linda          correct a speaker in the cast sheet
  cast S1>S2 self_term=mẹ other_term=con    set how S1 and S2 address each other
"""


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
    elif act == "cast":
        # cast S2 gender=female name=Linda      |   cast S1>S2 self_term=mẹ other_term=con
        value: dict = {}
        if ">" in unit:
            value["speaker"], value["listener"] = unit.split(">", 1)
            value["lang"] = a.tgt
        else:
            value["speaker"] = unit
        for kv in args[1:]:
            k, _, v = kv.partition("=")
            value[k.strip()] = int(v) if k.strip() == "since_ms" else v.strip()
        e = store.add("cast", "cast", value)
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


def cmd_gc(a) -> int:
    if a.remote:
        from dub import remote

        state = remote.PodState.load(a.workspace)
        if state is None:
            print("no pod is running (dub up first)", file=sys.stderr)
            return 2
        evict = a.older_than_hours * 3600 if a.older_than_hours is not None else None
        rep = remote.Dubd(state.url, state.token).gc(a.min_age_min * 60, a.dry_run, evict)
    else:
        from dub.store.gc import collect

        evict = a.older_than_hours * 3600 if a.older_than_hours is not None else None
        rep = collect(Workspace.open(a.workspace), min_age_s=a.min_age_min * 60,
                      dry_run=a.dry_run, evict_older_than_s=evict).as_dict()  # fmt: skip
    verb = "would free" if a.dry_run else "freed"
    if rep.get("evicted"):
        print(f"forgot {rep['evicted']} stored audio results")
    print(f"{verb} {rep['freed_mb']} MB: {rep['deleted']} files deleted, {rep['kept']} kept, "
          f"{rep['tmp_deleted']} temp files")  # fmt: skip
    for e in rep["errors"]:
        print("  error:", e)
    return 0


def cmd_bench_speed(a) -> int:
    from dub.bench.speed import analyse, render

    if a.target.is_dir():
        job = a.target
    else:  # a media file: dub it first (cached stages are reused, so bench a fresh run)
        opts = RunOptions(
            video=a.target, tgt=a.tgt, src=a.src, profile=a.profile, remote=a.remote,
            fake=a.fake, config_dir=a.config_dir, use=a.use, fresh=a.fresh, params=a.param,
        )  # fmt: skip
        job = run_video(opts, Workspace.open(a.workspace), log=print).job_dir
    media_s = None
    if (job / "tiles.json").exists():
        tiles = json.loads((job / "tiles.json").read_text(encoding="utf-8"))
        items = tiles.get("tiles", tiles) if isinstance(tiles, dict) else tiles
        media_s = max((t["end_ms"] for t in items), default=0) / 1000 or None
    rep = analyse(job, media_s)
    print(render(rep))
    if a.json:
        a.json.write_text(json.dumps(rep.as_dict(), indent=1), encoding="utf-8")
    return 0


def cmd_bench(a) -> int:
    from dub.bench.seg import boundary_f1

    if a.what == "speed":
        return cmd_bench_speed(a)
    if a.golden is None:
        print("error: bench seg needs --golden", file=sys.stderr)
        return 2
    if a.target.is_dir() or a.target.suffix == ".json":
        job = a.target if a.target.is_dir() else a.target.parent
    else:  # a media file: run the analysis (phase A) first
        opts = RunOptions(
            video=a.target, tgt="und", src=a.src, remote=a.remote, fake=a.fake,
            analysis_only=True, config_dir=a.config_dir, use=a.use, params=a.param,
        )  # fmt: skip
        job = run_video(opts, Workspace.open(a.workspace), log=print).job_dir
    seg = json.loads((job / "segmentation.json").read_text(encoding="utf-8"))
    scores: dict = {"job": str(job), "lang": seg["lang"]}
    gold = [ln.strip() for ln in a.golden.read_text(encoding="utf-8").splitlines() if ln.strip()]
    pred = [s["text"] for s in seg["sentences"]]
    lang = seg["lang"]
    for tol in sorted({0, a.tol}):
        f = boundary_f1(pred, gold, lang, tol=tol)
        scores[f"tol{tol}"] = f.as_dict()
        print(
            f"sentences (tolerance {tol} word): P {f.precision:.3f}  R {f.recall:.3f}  "
            f"F1 {f.f1:.3f}   ours {f.n_pred}, reference {f.n_gold}"
        )
    lens = [(u["end_ms"] - u["start_ms"]) / 1000 for u in seg["units"]]
    print(
        f"units: {len(lens)}, longest {max(lens, default=0):.1f} s, "
        f"over 15 s: {sum(1 for x in lens if x > 15)}, under 1 s: {sum(1 for x in lens if x < 1)}"
    )
    scores["units"] = len(lens)
    scores["unit_s_max"] = round(max(lens, default=0), 2)
    if a.json:
        a.json.write_text(json.dumps(scores, indent=1), encoding="utf-8")
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
        "bench": cmd_bench,
        "gc": cmd_gc,
    }[a.cmd]
    return handler(a)


if __name__ == "__main__":
    raise SystemExit(main())
