"""`dub get` with no link: a page on this computer where you paste links and download them.

The page lives at http://127.0.0.1:8766/ and talks to this server only. Downloads run one
at a time in a background thread; the page polls /api/state for progress. The server
answers only requests addressed to 127.0.0.1/localhost with no foreign Origin, so another
web site open in the browser cannot start downloads through it.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dub.fetch import FetchError, fetch, is_url

PAGE = Path(__file__).with_name("page.html")
MAX_URLS = 50


@dataclass
class Item:
    id: int
    url: str
    status: str = "queued"  # queued | downloading | done | error
    pct: float | None = None
    speed: str = ""
    eta: str = ""
    title: str = ""
    site: str = ""
    path: str = ""
    size_mb: float | None = None
    duration_s: float | None = None
    error: str = ""
    hint: str = ""


class FetchApp:
    def __init__(self, out_dir: Path, cookies: Path | None = None,
                 cookies_from_browser: str | None = None):  # fmt: skip
        self.out_dir = out_dir
        self.cookies = cookies
        self.cookies_from_browser = cookies_from_browser
        self.items: list[Item] = []
        self.lock = threading.Lock()
        self.jobs: queue.Queue[tuple[Item, str | None]] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def add(self, urls: list[str], browser: str | None) -> list[int]:
        ids = []
        with self.lock:
            for url in urls[:MAX_URLS]:
                item = Item(id=len(self.items) + 1, url=url)
                self.items.append(item)
                ids.append(item.id)
                self.jobs.put((item, browser))
        return ids

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "out_dir": str(self.out_dir),
                "cookies_from_browser": self.cookies_from_browser or "",
                "cookies_file": str(self.cookies or ""),
                "items": [asdict(i) for i in self.items],
            }

    def item(self, item_id: int) -> Item | None:
        with self.lock:
            return next((i for i in self.items if i.id == item_id), None)

    def _worker(self) -> None:
        while True:
            item, browser = self.jobs.get()
            self._download(item, browser)

    def _download(self, item: Item, browser: str | None) -> None:
        def progress(d: dict) -> None:
            with self.lock:
                if d.get("status") == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate")
                    done = d.get("downloaded_bytes") or 0
                    item.pct = round(100 * done / total, 1) if total else None
                    item.speed = _rate(d.get("speed"))
                    item.eta = _eta(d.get("eta"))
                elif d.get("status") == "finished":
                    item.pct = 100.0
                    item.eta = ""
                info = d.get("info_dict") or {}
                item.title = item.title or str(info.get("title") or "")

        with self.lock:
            item.status = "downloading"
        try:
            got = fetch(
                item.url, self.out_dir, cookies=self.cookies,
                cookies_from_browser=browser or self.cookies_from_browser, progress=progress,
            )  # fmt: skip
        except FetchError as e:
            with self.lock:
                item.status, item.error, item.hint = "error", str(e), e.hint
            return
        except Exception as e:  # keep the worker alive whatever happens
            with self.lock:
                item.status, item.error = "error", f"{type(e).__name__}: {e}"
            return
        with self.lock:
            item.status = "done"
            item.pct = 100.0
            item.title = got.title
            item.site = got.site
            item.path = str(got.path)
            item.duration_s = got.duration_s
            item.size_mb = round(got.path.stat().st_size / 1e6, 1)


def _rate(bps: float | None) -> str:
    return f"{bps / 1e6:.1f} MB/s" if bps else ""


def _eta(s: float | None) -> str:
    if s is None:
        return ""
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def show_in_folder(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.Popen(["explorer", f"/select,{path}"])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])


def make_handler(app: FetchApp, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
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
            self._send(code, json.dumps(body, ensure_ascii=False).encode(), "application/json")

        def _trusted(self) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:
                return False  # DNS rebinding: a foreign name pointed at 127.0.0.1
            origin = self.headers.get("Origin")
            return origin is None or origin.split("//", 1)[-1] in allowed_hosts

        def do_GET(self):
            if not self._trusted():
                return self._json(403, {"error": "forbidden"})
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._json(200, app.state())
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._trusted():
                return self._json(403, {"error": "forbidden"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json(415, {"error": "send JSON"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(min(n, 1_000_000)) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad JSON"})
            if self.path == "/api/get":
                text = str(body.get("urls") or "")
                urls = [u.strip() for u in text.split() if u.strip()]
                bad = [u for u in urls if not is_url(u)]
                urls = [u for u in urls if is_url(u)]
                if not urls:
                    return self._json(400, {"error": "no link starting with https://", "bad": bad})
                browser = str(body.get("browser") or "").strip() or None
                return self._json(200, {"ids": app.add(urls, browser), "bad": bad})
            if self.path == "/api/show":
                item = app.item(int(body.get("id") or 0))
                if item is None or not item.path:  # only files this server downloaded
                    return self._json(404, {"error": "no such download"})
                show_in_folder(Path(item.path))
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})

    return Handler


def serve_fetch(
    out_dir: Path,
    *,
    port: int = 8766,
    open_browser: bool = True,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
) -> int:
    app = FetchApp(out_dir, cookies, cookies_from_browser)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port))
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"[get] {url}  saving to {out_dir}  (Ctrl+C to stop)", flush=True)
    if open_browser and not os.environ.get("DUB_NO_BROWSER"):
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
