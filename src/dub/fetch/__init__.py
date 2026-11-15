"""Download a video from a link (YouTube, TikTok, Facebook, X, Bilibili, ... ) with yt-dlp.

`dub get URL` saves the video next to your other downloads; `dub run URL --to vi` downloads
it first and dubs the file. The work is done by yt-dlp (the `fetch` dependency group),
which knows well over a thousand sites; this module picks the format, names the file and
turns yt-dlp's errors into a hint you can act on.

Some sites only answer a browser that has visited them (Douyin, Instagram, private or
age-restricted posts): pass the cookies of your own browser with `cookies_from_browser`
("firefox", "chrome", "edge", ...) or a `cookies.txt` file exported from it. On Windows,
Chrome and Edge encrypt their cookies so that other programs cannot read them; Firefox, or
a cookies.txt export, works there.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

URL_RE = re.compile(r"^https?://\S+$", re.I)
# H.264 + AAC first, then the highest resolution: every player plays H.264, while Windows
# needs a paid extension for H.265, the only codec of TikTok's 1080p (the dub keeps the
# video stream as it is). Measured: YouTube still gives 1080p H.264, TikTok 720p H.264.
FORMAT_SORT = ["vcodec:h264", "res", "acodec:aac"]
FORMAT = "bv*+ba/b"
TITLE_BYTES = 70  # file names keep the start of the title (TikTok titles are whole captions)
JS_RUNTIMES = ("deno", "node", "bun")  # YouTube needs one to solve its player challenge


class FetchError(RuntimeError):
    """A download failed; `hint` says what to do about it, when we know."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


@dataclass
class Fetched:
    path: Path
    url: str
    site: str
    video_id: str
    title: str
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    vcodec: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def is_url(text: str) -> bool:
    return bool(URL_RE.match(text.strip()))


def default_out_dir() -> Path:
    return Path.home() / "Downloads"


def js_runtimes() -> dict[str, dict]:
    """The JavaScript runtimes found on PATH, in yt-dlp's option format."""
    return {name: {} for name in JS_RUNTIMES if shutil.which(name)}


# First match wins: the cookie-database errors also mention cookies.
HINTS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"DPAPI|decrypt|app.bound|could not copy .*cookie database", re.I),
        "this browser encrypts its cookies so that other programs cannot read them (Chrome "
        "and Edge on Windows do): use Firefox, or export a cookies.txt and pass --cookies FILE",
    ),
    (
        re.compile(r"cookies|log ?in|sign in|age[- ]restricted|confirm your age", re.I),
        "the site only shows this post to a browser that has visited it (or is logged in): "
        "open the link once in Firefox and pass --cookies-from-browser firefox, or export a "
        "cookies.txt from your browser and pass --cookies FILE",
    ),
    (
        re.compile(r"IP address is blocked|not available in your (country|region)|geo", re.I),
        "the post is not available here: it was deleted, made private, or is not shown in "
        "your region (TikTok gives the same answer for all three)",
    ),
    (
        re.compile(r"Unsupported URL", re.I),
        "yt-dlp does not know this site; check the link, or download the video yourself "
        "and give `dub run` the file",
    ),
    (
        re.compile(r"\bprivate\b", re.I),
        "the post is private: only an account that may see it can download it (pass that "
        "browser's cookies)",
    ),
    (
        re.compile(r"ffmpeg", re.I),
        "ffmpeg is needed to join the video and audio of high-resolution formats: install "
        "it and make sure it is on PATH",
    ),
]


def hint_for(message: str) -> str:
    for pattern, hint in HINTS:
        if pattern.search(message):
            return hint
    return ""


def ascii_ca_bundle() -> str | None:
    """curl-cffi (the browser-like client TikTok needs) cannot open a CA bundle whose path has
    non-ASCII letters on Windows (a project under "E:/Project cá nhân/..."): every TikTok
    request then fails as "HTTP Error 0". Copy certifi's bundle to an ASCII folder and point
    SSL_CERT_FILE at it, before curl-cffi is imported. Returns the path set, if any."""
    if os.environ.get("SSL_CERT_FILE"):
        return None
    try:
        import certifi
    except ImportError:  # pragma: no cover - comes with yt-dlp[default]
        return None
    src = certifi.where()
    if src.isascii():
        return None
    for folder in (tempfile.gettempdir(), os.environ.get("PUBLIC", ""), "C:/Users/Public"):
        if not folder or not folder.isascii() or not os.path.isdir(folder):
            continue
        dst = Path(folder) / "dub-cacert.pem"
        try:
            if not dst.exists() or dst.read_bytes() != Path(src).read_bytes():
                shutil.copyfile(src, dst)
        except OSError:
            continue
        os.environ["SSL_CERT_FILE"] = str(dst)
        return str(dst)
    return None


def _yt_dlp():
    ascii_ca_bundle()
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover - depends on the install
        raise FetchError(
            "yt-dlp is not installed",
            "install the download tools with `uv sync --group fetch`",
        ) from e
    return yt_dlp


def ydl_options(
    out_dir: Path,
    *,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
    progress: Callable[[dict], None] | None = None,
    quiet: bool = True,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "format": FORMAT,
        "format_sort": FORMAT_SORT,
        "merge_output_format": "mp4",
        "paths": {"home": str(out_dir)},
        "outtmpl": {"default": f"%(title).{TITLE_BYTES}B [%(id)s].%(ext)s"},
        "noplaylist": True,  # a link to one video inside a playlist downloads that video
        "windowsfilenames": True,
        "quiet": quiet,
        "no_warnings": quiet,
        "noprogress": quiet,
        "retries": 5,
        "fragment_retries": 5,
        "continuedl": True,
    }
    runtimes = js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes
    if cookies is not None:
        opts["cookiefile"] = str(cookies)
    if cookies_from_browser:
        browser, _, profile = cookies_from_browser.partition(":")
        opts["cookiesfrombrowser"] = (browser.lower(), profile or None, None, None)
    if progress is not None:
        opts["progress_hooks"] = [progress]
    return opts


def fetch(
    url: str,
    out_dir: Path | None = None,
    *,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
    progress: Callable[[dict], None] | None = None,
    quiet: bool = True,
) -> Fetched:
    """Download one video and return where it is. A file already downloaded is reused."""
    url = url.strip()
    if not is_url(url):
        raise FetchError(f"not a link: {url!r}", "give a full link starting with https://")
    out_dir = (out_dir or default_out_dir()).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    yt_dlp = _yt_dlp()
    opts = ydl_options(
        out_dir, cookies=cookies, cookies_from_browser=cookies_from_browser,
        progress=progress, quiet=quiet,
    )  # fmt: skip
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        msg = _clean(str(e))
        raise FetchError(msg, hint_for(msg)) from e
    except Exception as e:  # cookie database errors and the like are not DownloadErrors
        msg = _clean(str(e)) or type(e).__name__
        raise FetchError(msg, hint_for(msg)) from e
    if info is None:
        raise FetchError(f"nothing to download at {url}")
    if info.get("_type") == "playlist":  # a profile or playlist link: noplaylist does not apply
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise FetchError(f"no video found at {url}")
        info = entries[0]
    path = _final_path(info)
    if path is None or not path.exists():
        raise FetchError(f"yt-dlp finished but the file is missing ({path})")
    return Fetched(
        path=path,
        url=url,
        site=str(info.get("extractor_key") or info.get("extractor") or ""),
        video_id=str(info.get("id") or ""),
        title=str(info.get("title") or ""),
        duration_s=info.get("duration"),
        width=info.get("width"),
        height=info.get("height"),
        vcodec=str(info.get("vcodec") or ""),
    )


def _final_path(info: dict) -> Path | None:
    for d in info.get("requested_downloads") or []:
        p = d.get("filepath") or d.get("_filename")
        if p:
            return Path(p)
    p = info.get("filepath") or info.get("_filename")
    return Path(p) if p else None


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(message: str) -> str:
    message = ANSI_RE.sub("", message).strip()
    return re.sub(r"^ERROR:\s*", "", message)
