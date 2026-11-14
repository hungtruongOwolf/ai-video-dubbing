"""Sentence segmentation on real videos: the pipeline vs the sentences really spoken.

Videos with human subtitles in the spoken language, from Wikimedia Commons (public domain or
free licences). The reference ("golden") sentences come from those subtitles:
- en, es, vi: the subtitles are punctuated; sentences are split at their punctuation
  (abbreviations and initials handled by the same rules the pipeline falls back to).
- zh, ja: Commons subtitles separate phrases with spaces and have no sentence punctuation.
  The transcript of the first minutes was punctuated by hand (by Claude, the assistant that
  wrote this pipeline) and is stored in experiments/golden/<id>.txt. Treat these as silver
  labels: human words, assistant-made sentence ends.

Only the first `minutes` of each video are used (the design's "golden" set: first 5 minutes).

Steps (from the repo root, on the laptop; the analysis runs on the GPU pod):
    uv run python experiments/real_videos.py fetch     # download, cut, build references
    uv run python experiments/real_videos.py run       # dub bench seg ... --remote per video
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, save_json  # noqa: E402

UA = {"User-Agent": "dub-experiments/0.1 (personal research script; low volume)"}
GOLDEN = Path(__file__).parent / "golden"

# id, language, Commons file, minutes used, kind of speech
VIDEOS = [
    ("en-obama-2016", "en", "2016-08-20 President Obama's Weekly Address.webm", 5, "address"),
    ("en-obama-2009", "en", "20090124 WeeklyAddress.ogv", 5, "address"),
    ("en-jfk-rice", "en", "President Kennedy's Speech at Rice University.ogv", 6, "speech, crowd"),
    ("es-maduro", "es", 'Nicolás Maduro "De aquí no me saca nadie".webm', 5, "spontaneous"),
    ("es-wiki-cuenta", "es", "¿Cómo crear una cuenta en Wikipedia?.webm", 5, "tutorial"),
    (
        "vi-hcm-1945",
        "vi",
        "Ho Chi Minh reading the Proclamation of Independence of the "
        "Democratic Republic of Vietnam on 2 September 1945.wav",
        7,
        "1945 recording",
    ),
    ("vi-pho-lai", "vi", "Pho Lai Pagoda Buddhist Activities Summary.webm", 5, "documentary"),
    (
        "zh-vivo-interview",
        "zh",
        "采访vivo & MediaTek研发大佬：蓝厂与天玑合作背后的故事.webm",
        5,
        "interview",
    ),
    ("zh-polio", "zh", "Zh.Video-小兒麻痺症.webm", 5, "narration"),
    (
        "ja-digital-q",
        "ja",
        "デジタル監から東京都副知事に３つの質問「都のDX」「ユーザー視点」「キャリア改革」.webm",
        5,
        "conversation",
    ),
    ("ja-mynaportal", "ja", "初潜入。マイナポータル開発の舞台裏へ.webm", 5, "documentary"),
]
LABELLED = {"zh-vivo-interview", "ja-digital-q", "ja-mynaportal"}  # silver labels by hand


def _api(**kw) -> dict:
    q = urllib.parse.urlencode({**kw, "format": "json"})
    req = urllib.request.Request("https://commons.wikimedia.org/w/api.php?" + q, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(url: str) -> bytes:
    """GET with polite retries: Wikimedia answers 429 when downloads come too fast."""
    import time
    import urllib.error

    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            wait = float(e.headers.get("Retry-After") or 0) or 30 * (attempt + 1)
            print(f"   rate limited, waiting {wait:.0f} s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _smallest_with_audio(title: str) -> tuple[str, str]:
    """URL and extension of the lightest copy that still has the audio track."""
    r = _api(action="query", titles="File:" + title, prop="videoinfo",
             viprop="derivatives|url")  # fmt: skip
    info = next(iter(r["query"]["pages"].values()))["videoinfo"][0]
    best = (None, 1 << 62, "")
    for d in info.get("derivatives", []):
        key = d.get("transcodekey", "")
        if key in ("ogg", "mp3") or key.startswith(("144p", "240p", "360p")):
            rank = {"ogg": 0, "mp3": 1}.get(key, 2 + int(d.get("width") or 9999))
            if rank < best[1]:
                best = (d["src"], rank, "." + key.split(".")[-1])
    if best[0]:
        return best[0], best[2]
    return info["url"], Path(urllib.parse.urlparse(info["url"]).path).suffix


def _srt(title: str, lang: str) -> list[tuple[float, float, str]]:
    url = "https://commons.wikimedia.org/w/index.php?" + urllib.parse.urlencode(
        {"title": f"TimedText:{title}.{lang}.srt", "action": "raw"}
    )
    text = _get(url).decode("utf-8", "replace").replace("\r", "")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", block)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        body = block[m.end() :].strip()
        cues.append((a, b, " ".join(body.split())))
    return cues


def _clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|<[^>]+>", " ", text)  # [Applause], (Cheers), tags
    text = re.sub(r"^[A-Z][A-Za-z .]{2,40}:\s", "", text)  # "President Obama: "
    return " ".join(text.split())


def sentences_from_punctuation(text: str, lang: str) -> list[str]:
    from dub.contracts.v1 import Word, join_words
    from dub.ops.segment import rule_probs

    if lang in ("zh", "ja"):  # no spaces: cut after the full-width sentence marks
        return [s for s in re.split(r"(?<=[。！？])", text.replace(" ", "")) if s.strip()]
    tokens = text.split()
    out, cur = [], []
    for tok, p in zip(tokens, rule_probs(tokens), strict=True):
        cur.append(Word(w=tok, start_ms=0, end_ms=0))
        if p >= 0.5:
            out.append(join_words(cur, " "))
            cur = []
    if cur:
        out.append(join_words(cur, " "))
    return out


def fetch() -> None:
    from dub.ops.media import run_ffmpeg

    root = data_dir() / "realvid"
    root.mkdir(parents=True, exist_ok=True)
    index = {}
    for vid, lang, title, minutes, kind in VIDEOS:
        cues = _srt(title, lang)
        limit = minutes * 60
        kept = [c for c in cues if c[1] <= limit]
        end = kept[-1][1] + 0.5 if kept else limit
        src_url, ext = _smallest_with_audio(title)
        src = root / f"{vid}.src{ext}"
        if not src.exists():
            src.write_bytes(_get(src_url))
        cut = root / f"{vid}.mka"
        run_ffmpeg(["-i", str(src), "-map", "0:a:0", "-t", f"{end:.2f}", "-c:a", "copy",
                    "-map_metadata", "-1", "-fflags", "+bitexact", str(cut)])  # fmt: skip
        transcript = " ".join(_clean(c[2]) for c in kept)
        (root / f"{vid}.transcript.txt").write_text(transcript, encoding="utf-8")
        if vid in LABELLED:
            gold = GOLDEN / f"{vid}.txt"
        else:
            gold = root / f"{vid}.golden.txt"
            gold.write_text(
                "\n".join(sentences_from_punctuation(transcript, lang)), encoding="utf-8"
            )
        index[vid] = {"lang": lang, "title": title, "kind": kind, "seconds": round(end, 1),
                      "media": str(cut), "golden": str(gold)}  # fmt: skip
        print(vid, f"{end / 60:.1f} min", len(kept), "cues", flush=True)
    save_json(root / "index.json", index)


def run(
    params: list[str] | None = None,
    tag: str = "",
    use: list[str] | None = None,
    only: list[str] | None = None,
) -> None:
    """Score every video; `params` (--param overrides) and `tag` compare a variant, e.g.
    batched Whisper: run --param whisper_large_v3.batch_size=16 --tag batched."""
    from dub.cli import main as dub

    root = data_dir() / "realvid"
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    results = {}
    for vid, v in index.items():
        if only and vid not in only and v["lang"] not in only:
            continue
        if not Path(v["golden"]).exists():
            print(vid, "no reference sentences yet, skipped")
            continue
        out = root / f"{vid}.scores{('.' + tag) if tag else ''}.json"
        extra = [x for p in params or [] for x in ("--param", p)]
        extra += [x for u in use or [] for x in ("--use", u)]
        code = dub(["bench", "seg", v["media"], "--golden", v["golden"], "--from", v["lang"],
                    "--remote", "--json", str(out), *extra])  # fmt: skip
        if code == 0:
            results[vid] = {**v, **json.loads(out.read_text(encoding="utf-8"))}
    save_json(data_dir() / "results" / f"real_videos{('_' + tag) if tag else ''}.json", results)
    for vid, r in results.items():
        t1 = r["tol1"]
        print(
            f"{vid:<20} {r['lang']}  F1 {t1['f1']:.3f}  P {t1['p']:.3f}  R {t1['r']:.3f}  "
            f"ours {t1['n_pred']} ref {t1['n_gold']}  units {r['units']} max {r['unit_s_max']} s"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "run"])
    ap.add_argument("--param", action="append", default=[], help="PROVIDER.KEY=VALUE")
    ap.add_argument("--tag", default="", help="suffix of the score files of a variant")
    ap.add_argument("--use", action="append", default=[], help="OP=PROVIDER")
    ap.add_argument("--only", default="", help="comma-separated video ids or languages")
    a = ap.parse_args()
    if a.step == "fetch":
        fetch()
    else:
        run(a.param, a.tag, a.use, [x for x in a.only.split(",") if x])


if __name__ == "__main__":
    main()
