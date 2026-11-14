"""The review page: a small local web app on the laptop (http://127.0.0.1:8765).

    dub review video.mp4 --to vi [--remote]

It lists the dubbed lines riskiest first. For each line you can:
- play the original and the dub side by side;
- fix the text, ask for a new take, change the speaker, move the slot, split, merge, lock;
- undo any edit.

Each change is appended to the edits file. "Render changes" re-runs the job; thanks to the
cache, only the touched lines (and their 30 s tiles) are redone. "Export video" also does the
final loudness pass and muxes the result next to the video.

Audio previews:
- the original comes straight from the local video file (ffmpeg);
- the dub is cut from the mixed tiles, locally or on the pod (POST /preview).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import traceback
import urllib.parse
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dub.edits import EditStore
from dub.ops.media import _bin
from dub.ops.preview import cut_mp3, segments_for
from dub.runner import RunOptions, RunOutput, audio_key_for, finish, run_video
from dub.store import Workspace

PAGE = Path(__file__).with_name("page.html")
PAD_MS = 600  # play a little before and after the line


class ReviewApp:
    def __init__(self, opts: RunOptions, ws: Workspace, out_dir: Path | None = None):
        self.opts = replace(opts, video=opts.video.resolve(), export=False)
        self.ws = ws
        self.out_dir = out_dir
        self.audio_key = audio_key_for(ws, self.opts.video)
        self.store = EditStore.for_job(ws.root, self.audio_key, opts.tgt)
        self.out: RunOutput | None = None
        self.rendered_edits: set[str] = set()
        self.render: dict[str, Any] = {
            "running": False,
            "log": [],
            "error": None,
            "export": False,
            "video": None,
            "seconds": None,
        }
        self.lock = threading.Lock()

    # ------------------------------------------------------------ rendering
    def start_render(self, export: bool) -> bool:
        with self.lock:
            if self.render["running"]:
                return False
            self.render = {
                "running": True,
                "log": [],
                "error": None,
                "export": export,
                "video": None,
                "seconds": None,
            }
        threading.Thread(target=self._render, args=(export,), daemon=True).start()
        return True

    def _log(self, line: str) -> None:
        print(line, flush=True)
        self.render["log"] = (self.render["log"] + [line])[-200:]

    def _render(self, export: bool) -> None:
        t0 = time.time()
        edits_now = {e.id for e in self.store.all()}
        try:
            out = run_video(replace(self.opts, export=export), self.ws, log=self._log)
            if export:
                video = finish(replace(self.opts, export=True), out, self.out_dir, log=self._log)
                self.render["video"] = str(video) if video else None
            self.out = out
            self.rendered_edits = edits_now
        except Exception as e:  # noqa: BLE001 - surfaced on the page
            traceback.print_exc()
            self.render["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.render["seconds"] = round(time.time() - t0, 1)
            self.render["running"] = False

    # ------------------------------------------------------------ state
    def state(self) -> dict[str, Any]:
        units: list[dict] = []
        if self.out and (self.out.job_dir / "units.json").exists():
            units = json.loads((self.out.job_dir / "units.json").read_text(encoding="utf-8"))
        edits = self.store.summary()
        nudges: dict[str, dict] = {}
        for e in edits:
            if e["field"] == "nudge":
                nudges[e["unit_id"]] = e["value"]
        for row in units:
            row["nudge"] = nudges.get(row["unit"]["id"], {"start_ms": 0, "end_ms": 0})
        pending = [e for e in self.store.all() if e.id not in self.rendered_edits]
        cast, metrics = None, {}
        if self.out and (self.out.job_dir / "cast.json").exists():
            cast = json.loads((self.out.job_dir / "cast.json").read_text(encoding="utf-8"))["cast"]
        if self.out and (self.out.job_dir / "metrics.json").exists():
            metrics = json.loads((self.out.job_dir / "metrics.json").read_text(encoding="utf-8"))
        return {
            "video": self.opts.video.name, "tgt": self.opts.tgt, "remote": self.opts.remote,
            "job": self.out.job_id if self.out else None,
            "calls": self.out.calls if self.out else 0,
            "cache_hits": self.out.cache_hits if self.out else 0,
            "warnings": self.out.warnings if self.out else [],
            "units": units, "edits": edits, "pending": len(pending),
            "render": self.render, "cast": cast, "metrics": metrics,
        }  # fmt: skip

    # ------------------------------------------------------------ audio
    def audio_dub(self, from_ms: int, to_ms: int) -> bytes:
        if not self.out:
            return b""
        segs = segments_for(self.out.tiles, from_ms, to_ms)
        if self.out.remote:
            from dub import remote

            state = remote.PodState.load(self.ws.root)
            if state is None:
                raise RuntimeError("no pod recorded: run `dub up`")
            return remote.Dubd(state.url, state.token).preview(segs, from_ms, to_ms)
        return cut_mp3(self.ws.cas, segs, from_ms, to_ms)

    def audio_orig(self, from_ms: int, to_ms: int) -> bytes:
        cmd = [_bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{from_ms / 1000:.3f}",
               "-to", f"{to_ms / 1000:.3f}", "-i", str(self.opts.video), "-vn", "-ac", "1",
               "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "-"]  # fmt: skip
        return subprocess.run(cmd, capture_output=True, check=False).stdout


def make_handler(app: ReviewApp):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, body: Any) -> None:
            self._send(code, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")  # fmt: skip

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if url.path == "/api/state":
                    return self._json(200, app.state())
                if url.path in ("/api/audio/dub", "/api/audio/orig"):
                    a = max(0, int(q["from"]) - PAD_MS)
                    b = int(q["to"]) + PAD_MS
                    fn = app.audio_dub if url.path.endswith("dub") else app.audio_orig
                    data = fn(a, b)
                    return self._send(200 if data else 404, data, "audio/mpeg")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._json(500, {"error": str(e)})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad JSON"})
            if self.path == "/api/edit":
                e = app.store.add(body["unit_id"], body["field"], body.get("value"))
                return self._json(200, json.loads(e.model_dump_json()))
            if self.path == "/api/undo":
                e = app.store.undo(body.get("edit_id"))
                return self._json(200, {"undone": e.value if e else None})
            if self.path == "/api/render":
                started = app.start_render(bool(body.get("export")))
                return self._json(200 if started else 409, {"started": started})
            self._json(404, {"error": "not found"})

    return Handler


def serve_review(opts: RunOptions, ws: Workspace, *, port: int = 8765, open_browser: bool = True,
                 out_dir: Path | None = None) -> int:  # fmt: skip
    app = ReviewApp(opts, ws, out_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"[review] {url}  (Ctrl+C to stop)", flush=True)
    app.start_render(export=False)  # first render: everything cached is instant
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
