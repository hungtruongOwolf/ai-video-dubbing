"""dub get / dub run URL: link handling, yt-dlp options, error hints, the local page."""

from __future__ import annotations

import http.client
import json
import threading
import time
import types
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import dub.fetch as fetch_mod
from dub.cli import _video_or_url
from dub.fetch import FetchError, fetch, hint_for, is_url, ydl_options
from dub.fetch import server as fetch_server


def test_links_and_files():
    assert is_url("https://www.tiktok.com/@nasa/video/7691727731800673549")
    assert is_url("  http://youtu.be/jNQXAC9IVRw ")
    assert not is_url("video.mp4")
    assert not is_url("C:/Users/me/Downloads/a.mp4")
    assert not is_url("https://a b")
    # A link must stay a string: Path() would fold https:// into https:/
    assert _video_or_url("https://youtu.be/x") == "https://youtu.be/x"
    assert _video_or_url("clip.mp4") == Path("clip.mp4")


def test_hints():
    douyin = "[Douyin] 7707508237452463370: Fresh cookies (not necessarily logged in) are needed"
    assert "cookies" in hint_for(douyin)
    tiktok = "[TikTok] 6748451240264420610: Your IP address is blocked from accessing this post"
    assert "region" in hint_for(tiktok)
    chrome = "Failed to decrypt with DPAPI. See https://github.com/yt-dlp/yt-dlp/issues/10927"
    assert "encrypts its cookies" in hint_for(chrome)
    assert "Firefox" in hint_for("Could not copy Chrome cookie database")
    assert "does not know this site" in hint_for("Unsupported URL: https://example.com/")
    # "webpage" and "message" must not look like an age gate or a login wall
    assert hint_for("Unable to download webpage: HTTP Error 500 (caused by message)") == ""


def test_options(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod.shutil, "which", lambda n: "/bin/node" if n == "node" else None)
    o = ydl_options(tmp_path, cookies_from_browser="Firefox:default-release")
    assert o["format_sort"][:2] == ["vcodec:h264", "res"]  # plays everywhere, then sharpest
    assert o["merge_output_format"] == "mp4"
    assert o["noplaylist"] is True
    assert o["paths"] == {"home": str(tmp_path)}
    assert o["js_runtimes"] == {"node": {}}  # YouTube's challenge needs a JS runtime
    assert o["cookiesfrombrowser"] == ("firefox", "default-release", None, None)
    o = ydl_options(tmp_path, cookies=tmp_path / "c.txt")
    assert o["cookiefile"] == str(tmp_path / "c.txt") and "cookiesfrombrowser" not in o
    monkeypatch.setattr(fetch_mod.shutil, "which", lambda n: None)
    assert "js_runtimes" not in ydl_options(tmp_path)  # yt-dlp keeps its own default


def test_ca_bundle_moved_out_of_a_non_ascii_path(tmp_path, monkeypatch):
    # curl-cffi on Windows cannot open "E:/Project cá nhân/.../cacert.pem" (TikTok: HTTP Error 0)
    import certifi

    src = tmp_path / "Project cá nhân" / "cacert.pem"
    src.parent.mkdir()
    src.write_text("CERTS")
    ascii_dir = tmp_path / "tmp"
    ascii_dir.mkdir()
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(certifi, "where", lambda: str(src))
    monkeypatch.setattr(fetch_mod.tempfile, "gettempdir", lambda: str(ascii_dir))
    got = fetch_mod.ascii_ca_bundle()
    assert got == str(ascii_dir / "dub-cacert.pem") and got.isascii()
    assert Path(got).read_text() == "CERTS"
    assert fetch_mod.os.environ["SSL_CERT_FILE"] == got
    # already set by the user, or an ASCII path: left alone
    assert fetch_mod.ascii_ca_bundle() is None
    monkeypatch.delenv("SSL_CERT_FILE")
    monkeypatch.setattr(certifi, "where", lambda: str(ascii_dir / "dub-cacert.pem"))
    assert fetch_mod.ascii_ca_bundle() is None and "SSL_CERT_FILE" not in fetch_mod.os.environ


class _DownloadError(Exception):
    pass


def fake_yt_dlp(behaviour):
    """A stand-in for the yt_dlp module: `behaviour(url, opts)` returns the info dict."""

    class YoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            return behaviour(url, self.opts)

    return types.SimpleNamespace(
        YoutubeDL=YoutubeDL, utils=types.SimpleNamespace(DownloadError=_DownloadError)
    )


def test_fetch_returns_the_file(tmp_path, monkeypatch):
    def ok(url, opts):
        f = Path(opts["paths"]["home"]) / "NASA Minute [7691727731800673549].mp4"
        f.write_bytes(b"x" * 10)
        for hook in opts.get("progress_hooks", []):
            hook({"status": "finished", "info_dict": {"title": "NASA Minute"}})
        return {
            "id": "7691727731800673549", "title": "NASA Minute", "extractor_key": "TikTok",
            "duration": 61.0, "width": 1080, "height": 1920, "vcodec": "h265",
            "requested_downloads": [{"filepath": str(f)}],
        }  # fmt: skip

    monkeypatch.setattr(fetch_mod, "_yt_dlp", lambda: fake_yt_dlp(ok))
    seen = []
    got = fetch("https://www.tiktok.com/@nasa/video/7691727731800673549", tmp_path / "out",
                progress=seen.append)  # fmt: skip
    assert got.path.name == "NASA Minute [7691727731800673549].mp4" and got.path.exists()
    assert (got.site, got.video_id, got.duration_s, got.height) == ("TikTok", "7691727731800673549",
                                                                   61.0, 1920)  # fmt: skip
    assert seen and seen[0]["status"] == "finished"


def test_fetch_playlist_link_takes_the_first_video(tmp_path, monkeypatch):
    def playlist(url, opts):
        f = Path(opts["paths"]["home"]) / "first [a1].mp4"
        f.write_bytes(b"x")
        return {"_type": "playlist", "entries": [None, {"id": "a1", "title": "first",
                                                        "filepath": str(f)}]}  # fmt: skip

    monkeypatch.setattr(fetch_mod, "_yt_dlp", lambda: fake_yt_dlp(playlist))
    assert fetch("https://www.youtube.com/@someone", tmp_path).video_id == "a1"


def test_fetch_errors_carry_a_hint(tmp_path, monkeypatch):
    def refuse(url, opts):
        raise _DownloadError(
            "ERROR: [Douyin] 77: Fresh cookies (not necessarily logged in) are needed"
        )

    monkeypatch.setattr(fetch_mod, "_yt_dlp", lambda: fake_yt_dlp(refuse))
    with pytest.raises(FetchError) as e:
        fetch("https://www.douyin.com/video/77", tmp_path)
    assert str(e.value).startswith("[Douyin]")  # "ERROR:" dropped
    assert "cookies-from-browser" in e.value.hint

    with pytest.raises(FetchError) as e:
        fetch("not a link", tmp_path)
    assert "https://" in e.value.hint

    def missing(url, opts):
        return {"id": "1", "requested_downloads": [{"filepath": str(tmp_path / "gone.mp4")}]}

    monkeypatch.setattr(fetch_mod, "_yt_dlp", lambda: fake_yt_dlp(missing))
    with pytest.raises(FetchError, match="missing"):
        fetch("https://x.com/a/status/1", tmp_path)


# ------------------------------------------------------------------ the page


@pytest.fixture
def page(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url, out_dir, **kw):
        calls.append((url, kw))
        if "douyin" in url:
            raise FetchError("[Douyin] Fresh cookies are needed", hint_for("cookies"))
        kw["progress"]({"status": "downloading", "downloaded_bytes": 5, "total_bytes": 10})
        f = out_dir / "clip [1].mp4"
        f.write_bytes(b"x" * 2_000_000)
        return fetch_mod.Fetched(path=f, url=url, site="Youtube", video_id="1", title="clip",
                                 duration_s=19.0)  # fmt: skip

    monkeypatch.setattr(fetch_server, "fetch", fake_fetch)
    shown = []
    monkeypatch.setattr(fetch_server, "show_in_folder", shown.append)
    app = fetch_server.FetchApp(tmp_path, cookies_from_browser="firefox")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), fetch_server.BaseHTTPRequestHandler)
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = fetch_server.make_handler(app, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield types.SimpleNamespace(port=port, app=app, calls=calls, shown=shown)
    httpd.shutdown()


def _req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Host": f"127.0.0.1:{port}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"null") if r.getheader("Content-Type", "").startswith(
        "application/json") else None  # fmt: skip


def _wait_done(app, n, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        items = app.state()["items"]
        if len(items) >= n and all(i["status"] in ("done", "error") for i in items):
            return items
        time.sleep(0.02)
    raise AssertionError(app.state())


def test_page_downloads_and_reports(page):
    code, res = _req(page.port, "POST", "/api/get", {
        "urls": "https://youtu.be/jNQXAC9IVRw\nnot-a-link\nhttps://www.douyin.com/video/77",
        "browser": "",
    })  # fmt: skip
    assert code == 200 and res["ids"] == [1, 2] and res["bad"] == ["not-a-link"]
    items = _wait_done(page.app, 2)
    ok, bad = items
    assert ok["status"] == "done" and ok["size_mb"] == 2.0 and ok["title"] == "clip"
    assert bad["status"] == "error" and "cookies" in bad["hint"]
    # no browser picked on the page: the one given to `dub get` is used
    assert page.calls[0][1]["cookies_from_browser"] == "firefox"
    code, _ = _req(page.port, "POST", "/api/show", {"id": 1})
    assert code == 200 and page.shown == [Path(ok["path"])]
    assert _req(page.port, "POST", "/api/show", {"id": 2})[0] == 404  # failed: no file
    code, state = _req(page.port, "GET", "/api/state")
    assert code == 200 and len(state["items"]) == 2
    assert _req(page.port, "GET", "/")[0] == 200


def test_page_refuses_other_sites(page):
    # another web site open in the browser must not start downloads
    code, _ = _req(page.port, "POST", "/api/get", {"urls": "https://youtu.be/x"},
                   {"Origin": "https://evil.example"})  # fmt: skip
    assert code == 403
    # DNS rebinding: a foreign name resolving to 127.0.0.1
    code, _ = _req(page.port, "GET", "/api/state", headers={"Host": f"evil.example:{page.port}"})
    assert code == 403
    # a plain form post (no JSON) is refused too
    c = http.client.HTTPConnection("127.0.0.1", page.port, timeout=5)
    c.request("POST", "/api/get", body=b"urls=https://youtu.be/x",
              headers={"Content-Type": "application/x-www-form-urlencoded"})  # fmt: skip
    assert c.getresponse().status == 415
    assert page.calls == []
    code, _ = _req(page.port, "POST", "/api/get", {"urls": "https://youtu.be/x"},
                   {"Origin": f"http://127.0.0.1:{page.port}"})  # fmt: skip
    assert code == 200
