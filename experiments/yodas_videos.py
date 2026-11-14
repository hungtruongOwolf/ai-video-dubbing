"""More real videos for the segmentation benchmark: YODAS2 (vi, zh).

Why: the real-video set of M3 (`real_videos.py`, Wikimedia Commons) has only two videos per
language for Vietnamese and Chinese, too few to decide the M7 trials (each helped one video
and hurt the other), and Commons has no more Vietnamese speech with subtitles.

Source: YODAS2 (espnet/yodas2 on Hugging Face, CC BY 3.0): YouTube videos published under
CC BY with the subtitles their uploaders wrote (the `vi000` / `zh000` subsets). Uploaded
subtitles are often translations (fan subtitles of English films), so every candidate is
checked before it counts: Whisper must hear the expected language, and the pipeline's own
transcript must match the subtitles (character error rate, CER).

Reference sentences: the subtitles' punctuation, split by the same rules as `real_videos.py`;
a hand-made file in experiments/golden/<id>.txt wins (silver labels, like zh/ja in M3).

Steps (pod, core env, workers running; wrap in pod/busy.sh so the pod stays up):
    python experiments/yodas_videos.py fetch           # shards -> first minutes, cues, golden
    python experiments/yodas_videos.py run [--param P.K=V] [--use OP=P] [--tag T] [--only ID,..]
Output: $DUB_EXP_DATA/yodas/<id>.wav|cues.json|golden.txt, results/yodas[_tag].json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_PYLIB = Path(os.environ.get("DUB_EXP_DATA", "")) / "pylib"  # experiment-only packages
if _PYLIB.is_dir():
    sys.path.insert(1, str(_PYLIB))

from experiments.common import data_dir, run_worker, save_json, worker_url  # noqa: E402

HF = "https://huggingface.co/datasets/espnet/yodas2/resolve/main/data"
UA = "dub-experiments/0.1 (research script; low volume)"
GOLDEN = Path(__file__).parent / "golden"
MINUTES = 5
MAX_CER = 0.35  # subtitles that differ more than this from the speech are not a transcript
MIN_REF = 15  # reference sentences in 5 minutes; fewer: the subtitles are not punctuated

# (YouTube id, language, YODAS2 subset, shard, kind). Chosen from the subsets' text files:
# native speakers (titles, content), sentence punctuation in the subtitles, >= 5 minutes.
CANDIDATES = [
    ("7cw7ARr0Woc", "vi", "vi000", "00000034", "monk answering a question, spontaneous"),
    ("1eyplFhZu5Y", "vi", "vi000", "00000010", "model essays read aloud"),
    ("3oIyhjGnVLA", "vi", "vi000", "00000015", "student vlog"),
    ("1sDqjIs_4rU", "vi", "vi000", "00000008", "report, narration"),
    ("6_HyZDSeLVY", "vi", "vi000", "00000028", "vlog"),
    ("2c8OawTG8TQ", "vi", "vi000", "00000014", "book talk"),
    ("c6vHKZ9Uho0", "vi", "vi000", "00000051", "radio essay"),
    ("7KJ6fpaw8vA", "vi", "vi000", "00000033", "podcast"),
    ("eremzE9UQi4", "vi", "vi000", "00000058", "unboxing vlog"),
    ("fGzlkXYo0hI", "zh", "zh000", "00000026", "political commentary"),
    ("v5DZN7rsLh4", "zh", "zh000", "00000010", "market analysis"),
    ("dgnjO1qSF3I", "zh", "zh000", "00000005", "tech talk (traditional characters)"),
    ("t5IDTECWpWM", "zh", "zh000", "00000030", "astrology talk"),
    ("lLOvGz4qZWU", "zh", "zh000", "00000007", "story narration (traditional characters)"),
]


def _root() -> Path:
    d = data_dir() / "yodas"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _curl(url: str, out: Path) -> None:
    subprocess.run(["curl", "-sfL", "-A", UA, "-o", str(out), url], check=True)


def _cues(text: dict[str, str]) -> list[tuple[float, float, str]]:
    """Subtitle cues (start s, end s, text) from YODAS2's '<id>-<n>-<start cs>-<end cs>' keys,
    without markup and without the repeated cues some uploads have."""
    cues = []
    for k, t in text.items():
        m = re.match(r".*-(\d+)-(\d+)-(\d+)$", k)
        if not m:
            continue
        t = re.sub(r"&lt;[^&]*&gt;|<[^>]+>|\[[^\]]*\]", " ", t.replace("&amp;", "&"))
        cues.append((int(m.group(2)) / 100, int(m.group(3)) / 100, " ".join(t.split())))
    cues.sort()
    out: list[tuple[float, float, str]] = []
    for c in cues:
        if c[2] and not (out and c[2] == out[-1][2]):
            out.append(c)
    return out


def fetch(only: set[str]) -> None:
    """Shards -> the first MINUTES of each candidate, its cues and reference sentences. Done
    when <id>.cues.json exists; a full-length /tmp/<id>.wav from an earlier try is reused."""
    from experiments.real_videos import sentences_from_punctuation

    root = _root()
    by_shard: dict[tuple[str, str], list[tuple]] = {}
    for c in CANDIDATES:
        if (not only or c[0] in only or c[1] in only) and not (root / f"{c[0]}.cues.json").exists():
            by_shard.setdefault((c[2], c[3]), []).append(c)
    for (subset, shard), cands in sorted(by_shard.items()):
        text_f = root / f"{subset}-{shard}.json"
        if not text_f.exists():
            _curl(f"{HF}/{subset}/text/{shard}.json", text_f)
        texts = {v["audio_id"]: v["text"] for v in json.loads(text_f.read_text(encoding="utf-8"))}
        tar = Path("/tmp") / f"yodas-{subset}-{shard}.tar.gz"  # container disk, not the volume
        missing = [c[0] for c in cands if not Path(f"/tmp/{c[0]}.wav").exists()]
        try:
            if missing:
                print(f"{subset}/{shard}: downloading audio for {', '.join(missing)}", flush=True)
                _curl(f"{HF}/{subset}/audio/{shard}.tar.gz", tar)
                for vid in missing:
                    subprocess.run(["tar", "xzf", str(tar), "--no-same-owner", "-C", "/tmp",
                                    f"./{vid}.wav"], check=True)  # fmt: skip
            for vid, lang, *_ in cands:
                cues = [c for c in _cues(texts[vid]) if c[1] <= MINUTES * 60]
                end = cues[-1][1] + 0.5
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"/tmp/{vid}.wav", "-t",
                                f"{end:.2f}", "-ac", "1", "-c:a", "pcm_s16le",
                                str(root / f"{vid}.wav")], check=True)  # fmt: skip
                Path(f"/tmp/{vid}.wav").unlink()
                transcript = " ".join(c[2] for c in cues)
                if lang == "zh":  # the rules split zh at full-width marks only
                    transcript = transcript.replace("?", "？").replace("!", "！")
                sents = sentences_from_punctuation(transcript, lang)
                (root / f"{vid}.golden.txt").write_text("\n".join(sents), encoding="utf-8")
                save_json(root / f"{vid}.cues.json", cues)
                print(f"  {vid} {lang} {end / 60:.1f} min, {len(cues)} cues, "
                      f"{len(sents)} sentences by punctuation", flush=True)  # fmt: skip
        finally:
            tar.unlink(missing_ok=True)


def _simplified(text: str) -> str:
    """Traditional -> simplified Chinese: Whisper writes Mandarin in simplified characters,
    so subtitles in traditional characters would count every differing character as an error
    (and break the boundary alignment). Needs opencc-python-reimplemented (Apache-2.0) on the
    path, e.g. installed with `uv pip install --target $DUB_EXP_DATA/pylib`."""
    from opencc import OpenCC

    return OpenCC("t2s").convert(text)


def _norm(text: str, lang: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return "".join(text.split()) if lang in ("zh", "ja") else " ".join(text.split())


def _detect_lang(wav: Path) -> str:
    """Language Whisper hears in the first 30 s."""
    from dub.ops.media import run_ffmpeg
    from dub.store import Workspace

    ws = Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                        db_path=Path("/tmp/exp.sqlite"))  # fmt: skip
    tmp = ws.cas.tmp_path(".wav")
    run_ffmpeg(["-i", str(wav), "-t", "30", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                str(tmp)])  # fmt: skip
    key = ws.cas.put_file(tmp, move=True)
    res = run_worker(
        "asr", "whisper_large_v3", "asr.transcribe", {"audio_key": key, "lang": "auto"}
    )
    ws.cas.path(key).unlink(missing_ok=True)
    return res.get("lang", "?")


def run(params: list[str], use: list[str], tag: str, only: set[str]) -> None:
    from dub.cli import main as dub
    from dub.core.config import load_config
    from dub.ops.align import cer

    cfg = load_config()
    envs = sorted({m.env for m in cfg.providers.values() if m.runtime == "http" and m.env})
    os.environ["DUB_WORKERS"] = json.dumps({e: worker_url(e) for e in envs})
    root = _root()
    ws_dir = Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws"
    results = {}
    for vid, lang, _subset, _shard, kind in CANDIDATES:
        wav = root / f"{vid}.wav"
        if (only and vid not in only and lang not in only) or not wav.exists():
            continue
        cues = json.loads((root / f"{vid}.cues.json").read_text(encoding="utf-8"))
        subs = " ".join(c[2] for c in cues)
        hand = GOLDEN / f"{vid}.txt"
        if hand.exists():  # sentence ends by hand: the words must be the subtitles, unchanged
            lines = hand.read_text(encoding="utf-8").splitlines()
            if _norm(" ".join(lines), lang) != _norm(subs, lang):
                print(f"{vid}: {hand} does not match the subtitles, skipped", flush=True)
                continue
        golden = hand if hand.exists() else root / f"{vid}.golden.txt"
        if lang == "zh":  # compare in simplified characters, like Whisper writes them
            conv = root / f"{vid}.golden.simplified.txt"
            conv.write_text(_simplified(golden.read_text(encoding="utf-8")), encoding="utf-8")
            golden, subs = conv, _simplified(subs)
        out = root / f"{vid}.scores{('.' + tag) if tag else ''}.json"
        extra = [x for p in params for x in ("--param", p)] + [x for u in use for x in ("--use", u)]
        code = dub(["bench", "seg", str(wav), "--golden", str(golden), "--from", lang,
                    "--workspace", str(ws_dir), "--json", str(out), *extra])  # fmt: skip
        if code != 0:
            continue
        r = json.loads(out.read_text(encoding="utf-8"))
        seg = json.loads((Path(r["job"]) / "segmentation.json").read_text(encoding="utf-8"))
        heard = " ".join(s["text"] for s in seg["sentences"])
        r["subs_cer"] = round(cer(_norm(subs, lang), _norm(heard, lang)), 3)
        lang_f = root / f"{vid}.heard_lang"
        if not lang_f.exists():
            lang_f.write_text(_detect_lang(wav), encoding="utf-8")
        r["heard_lang"] = lang_f.read_text(encoding="utf-8").strip()
        # unpunctuated subtitles give a handful of "sentences": no reference without hand labels
        r["valid"] = (
            r["heard_lang"] == lang and r["subs_cer"] <= MAX_CER and r["tol1"]["n_gold"] >= MIN_REF
        )
        r.update({"lang": lang, "kind": kind, "golden": golden.name})
        results[vid] = r
    save_json(data_dir() / "results" / f"yodas{('_' + tag) if tag else ''}.json", results)
    for vid, r in results.items():
        t1 = r["tol1"]
        print(f"{vid:<12} {r['lang']} {'ok ' if r['valid'] else 'BAD'} heard {r['heard_lang']} "
              f"CER {r['subs_cer']:.2f}  F1 {t1['f1']:.3f} P {t1['p']:.3f} R {t1['r']:.3f}  "
              f"ours {t1['n_pred']} ref {t1['n_gold']}  {r['kind']}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "run"])
    ap.add_argument("--param", action="append", default=[], help="PROVIDER.KEY=VALUE")
    ap.add_argument("--use", action="append", default=[], help="OP=PROVIDER")
    ap.add_argument("--tag", default="", help="suffix of the score files of a variant")
    ap.add_argument("--only", default="", help="comma-separated video ids or languages")
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}
    if a.step == "fetch":
        fetch(only)
    else:
        run(a.param, a.use, a.tag, only)


if __name__ == "__main__":
    main()
