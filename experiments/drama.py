"""Build the 4-person family scene of drama.json as audio, with the truth of who speaks when.

Each character is voiced by OmniVoice cloning a different public reference voice (F5-TTS
examples, MIT), at the English langpack speaking rate. Lines are joined with short, varied
pauses (some quick turn-takes of 150 ms), over a quiet music bed. The result is the test
material of experiments E8 (diarization) and E10 (forms of address, en -> vi).

On the pod (workers running):   envs/core/bin/python experiments/drama.py make
On the laptop:                  uv run python experiments/drama.py fetch   (download + video)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, run_worker, save_json  # noqa: E402

F5 = "https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/examples"
VOICE_FILES = {
    "country": f"{F5}/multi/country.flac",
    "main": f"{F5}/multi/main.flac",
    "town": f"{F5}/multi/town.flac",
    "zh_m": f"{F5}/basic/basic_ref_zh.wav",
}
GAPS_MS = [450, 300, 650, 150, 500, 800, 350, 150, 600, 400]  # deterministic, varied
SCRIPT = Path(__file__).parent / "drama.json"


def _workspace():
    from dub.store import Workspace

    return Workspace.open(
        Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws", db_path=Path("/tmp/exp.sqlite")
    )


def make() -> None:
    import numpy as np

    from dub.contracts.v1 import AudioReq, StretchReq, TrimReq
    from dub.ops import media
    from dub.ops.audio_io import read_wav, write_wav
    from dub.ops.synth import pad
    from dub.ops.timing import count_units

    cas = _workspace().cas
    script = json.loads(SCRIPT.read_text(encoding="utf-8"))
    refs = {}
    for name, url in VOICE_FILES.items():
        raw = data_dir() / f"voice_{name}{Path(url).suffix}"
        if not raw.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "dub-experiments"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw.write_bytes(r.read())
        out = cas.tmp_path(".wav")
        media.run_ffmpeg(["-i", str(raw), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
                          str(out)])  # fmt: skip
        key = cas.put_file(out, move=True)
        a16 = media.to_mono16k(AudioReq(audio_key=key), cas)
        tl = run_worker("asr", "whisper_large_v3", "asr.transcribe",
                        {"audio_key": a16.audio_key, "lang": "auto"})  # fmt: skip
        refs[name] = (key, " ".join(w["w"] for w in tl["words"]))
        print("voice", name, "->", refs[name][1][:60], flush=True)

    sr = 24000
    parts, truth, pos = [np.zeros(sr // 2, dtype=np.float32)], [], sr // 2
    for i, (spk, to, text) in enumerate(script["lines"]):
        voice = script["voices"][spk]
        target = round(count_units(text, "syllable") / 4.2 * 1000)
        res = run_worker("tts", "omnivoice", "tts.synthesize", {
            "unit_id": f"drama-{i}", "text": text, "lang": "en", "speaker": spk,
            "voice_key": refs[voice][0], "voice_text": refs[voice][1], "target_ms": target,
        })  # fmt: skip
        trimmed = media.trim(TrimReq(audio_key=res["audio_key"], pad_ms=0), cas)
        clip, _ = read_wav(cas.path(trimmed.audio_key))
        clip = clip[:, 0]
        truth.append({"line": i, "speaker": spk, "to": to, "text": text,
                      "onset_ms": round(pos * 1000 / sr),
                      "offset_ms": round((pos + len(clip)) * 1000 / sr)})  # fmt: skip
        gap = np.zeros(sr * GAPS_MS[i % len(GAPS_MS)] // 1000, dtype=np.float32)
        parts += [clip, gap]
        pos += len(clip) + len(gap)
    speech = np.concatenate(parts + [np.zeros(sr, dtype=np.float32)])
    tmp = cas.tmp_path(".wav")
    write_wav(tmp, speech, sr)
    voice48 = media.stretch(StretchReq(audio_key=cas.put_file(tmp, move=True), rate=1.0,
                                             sample_rate=48000), cas)  # fmt: skip
    v, _ = read_wav(cas.path(voice48.audio_key))
    bed = pad(len(v) * 1000 // 48000 + 1000, sr=48000, level=0.02)[: len(v)]
    mix = np.stack([v[:, 0] * 0.8 + bed, v[:, 0] * 0.8 + bed], axis=1)
    out = cas.tmp_path(".wav")
    write_wav(out, mix, 48000)
    key = cas.put_file(out, move=True)
    save_json(
        data_dir() / "results" / "drama.json",
        {"audio_key": key, "lines": truth, "seconds": round(len(v) / 48000, 1)},
    )
    print("drama", key, f"{len(v) / 48000:.1f} s", len(truth), "lines", flush=True)


def fetch() -> None:
    """Laptop: download the scene from the pod and wrap it in a video file."""
    from dub import remote
    from dub.ops.media import run_ffmpeg

    info = json.loads((data_dir() / "results" / "drama.json").read_text(encoding="utf-8"))
    ws_root = Path("dub-workspace")
    state = remote.PodState.load(ws_root)
    dubd = remote.Dubd(state.url, state.token)
    out_dir = data_dir() / "drama"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = dubd.download(f"/cas/{info['audio_key']}", out_dir / "drama.wav")
    video = out_dir / "drama.mp4"
    src = f"testsrc2=size=640x360:rate=25:duration={info['seconds'] + 1}"
    run_ffmpeg(["-f", "lavfi", "-i", src,
                "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                "-shortest", str(video)])  # fmt: skip
    (out_dir / "truth.json").write_text(json.dumps(info["lines"], ensure_ascii=False, indent=1),
                                        encoding="utf-8")  # fmt: skip
    print(video)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "fetch"])
    a = ap.parse_args()
    {"make": make, "fetch": fetch}[a.step]()


if __name__ == "__main__":
    main()
