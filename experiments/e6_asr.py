"""Experiment E6: which ASR per source language, and should it hear the mix or the vocals?

Material: the 11 real videos of real_videos.py (first ~5 minutes, human subtitles in the
spoken language). For each video the ASR runs on the original mix (16 kHz) and on the
separated vocals, with every candidate model for that language:
- whisper_large_v3 for all languages;
- phowhisper_large_ct2 for Vietnamese;
- (kotoba_whisper_v2 for Japanese was tried: faster-whisper cannot align its words, the
  distilled decoder has no alignment heads, and crashes; it cannot serve the pipeline.)
Score: CER against the subtitles (letters and digits only, case and punctuation ignored).
Subtitles are not always verbatim, so compare columns; the absolute level includes their
own deviations.

Laptop:  uv run python experiments/e6_asr.py prep        (uploads the references, prints a key)
Pod:     envs/core/bin/python experiments/e6_asr.py run --input KEY [--only ID,ID]
         envs/core/bin/python experiments/e6_asr.py analyse --input KEY   (vi: numbers, diffs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, run_worker, save_json  # noqa: E402

MODELS = {
    "en": ["whisper_large_v3"],
    "es": ["whisper_large_v3"],
    "vi": ["phowhisper_large_ct2", "whisper_large_v3"],
    "zh": ["whisper_large_v3"],
    "ja": ["whisper_large_v3"],
}


def prep() -> None:
    from dub import remote

    root = data_dir() / "realvid"
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    sources = json.loads(Path("dub-workspace/sources.json").read_text(encoding="utf-8"))
    by_path = {str(Path(k).resolve()): v["audio_key"] for k, v in sources.items()}
    items = []
    for vid, v in index.items():
        key = by_path.get(str(Path(v["media"]).resolve()))
        if not key:
            print("no uploaded audio for", vid)
            continue
        ref = (root / f"{vid}.transcript.txt").read_text(encoding="utf-8")
        items.append({"id": vid, "lang": v["lang"], "audio_key": key, "reference": ref})
    path = root / "e6_input.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    state = remote.PodState.load(Path("dub-workspace"))
    key = remote.Dubd(state.url, state.token).upload(path, ".json")
    print(key)


def run(input_key: str, only: list[str] | None = None) -> None:
    from dub.contracts.v1 import AudioReq, DecodeReq
    from dub.ops import media
    from dub.ops.align import cer
    from dub.store import Workspace

    ws = Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                        db_path=Path("/tmp/exp.sqlite"))  # fmt: skip
    cas = ws.cas
    items = [it for it in json.loads(cas.read_text(input_key)) if not only or it["id"] in only]
    out = data_dir() / "results" / "e6.json"
    out_text = data_dir() / "results" / "e6_text.json"  # what each model heard
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    old_text = json.loads(out_text.read_text(encoding="utf-8")) if out_text.exists() else {}
    results, texts = {}, {}
    for it in items:
        info = media.decode(DecodeReq(src_key=it["audio_key"]), cas)
        stems = run_worker("sep", "mel_roformer", "audio.separate", {"audio_key": info.a48k_key})
        vocals = media.to_mono16k(AudioReq(audio_key=stems["vocals_key"]), cas).audio_key
        row = {}
        for model in MODELS[it["lang"]]:
            for name, key in (("mix", info.a16k_key), ("vocals", vocals)):
                tl = run_worker(
                    "asr", model, "asr.transcribe", {"audio_key": key, "lang": it["lang"]}
                )
                joiner = "" if it["lang"] in ("zh", "ja") else " "
                heard = joiner.join(w["w"] for w in tl["words"])
                row[f"{model}/{name}"] = round(cer(it["reference"], heard), 4)
                texts[f"{it['id']}/{model}/{name}"] = heard
        results[it["id"]] = {"lang": it["lang"], **row}
        print(it["id"], row, flush=True)
        # saved after every video: a crash keeps what was measured
        save_json(out, {**old, **results})
        save_json(out_text, {**old_text, **texts})


VI_NUMBERS = set(
    "không một mốt hai ba bốn tư năm lăm sáu bảy bẩy tám chín mười mươi trăm nghìn ngàn "
    "triệu tỷ tỉ linh lẻ".split()
)


def _words(text: str) -> list[str]:
    import re
    import unicodedata

    return re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())


def analyse(input_key: str) -> None:
    """Where the Vietnamese errors are: CER again with numbers taken out of both sides
    (digits in the subtitles vs words in the ASR, or the reverse), and the largest
    word-level differences."""
    import difflib

    from dub.ops.align import cer
    from dub.store import Workspace

    ws = Workspace.open(Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws",
                        db_path=Path("/tmp/exp.sqlite"))  # fmt: skip
    refs = {it["id"]: it["reference"] for it in json.loads(ws.cas.read_text(input_key))}
    texts = json.loads((data_dir() / "results" / "e6_text.json").read_text(encoding="utf-8"))

    def strip_numbers(ws_: list[str]) -> str:
        return " ".join(w for w in ws_ if w not in VI_NUMBERS and not any(c.isdigit() for c in w))

    out = {}
    for name, heard in sorted(texts.items()):
        vid = name.split("/")[0]
        if not vid.startswith("vi-"):
            continue
        rw, hw = _words(refs[vid]), _words(heard)
        row = {
            "cer": round(cer(refs[vid], heard), 4),
            "cer_no_numbers": round(cer(strip_numbers(rw), strip_numbers(hw)), 4),
            "ref_words": len(rw),
            "heard_words": len(hw),
        }
        sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
        chunks = [(max(i2 - i1, j2 - j1), tag, " ".join(rw[i1:i2]), " ".join(hw[j1:j2]))
                  for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]  # fmt: skip
        chunks.sort(reverse=True)
        row["largest_differences"] = [
            {"op": t, "ref": r[:160], "heard": h[:160]} for _, t, r, h in chunks[:6]
        ]
        out[name] = row
        print(name, {k: v for k, v in row.items() if k != "largest_differences"})
        for c in row["largest_differences"]:
            print("   ", c["op"], "| ref:", c["ref"][:90], "| heard:", c["heard"][:90])
    save_json(data_dir() / "results" / "e6_vi.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "run", "analyse"])
    ap.add_argument("--input")
    ap.add_argument("--only", default="", help="comma-separated video ids")
    a = ap.parse_args()
    if a.step == "prep":
        prep()
    elif a.step == "analyse":
        analyse(a.input)
    else:
        run(a.input, [x for x in a.only.split(",") if x])


if __name__ == "__main__":
    main()
