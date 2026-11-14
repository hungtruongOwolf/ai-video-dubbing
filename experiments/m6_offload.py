"""M6 experiment: free GPU memory by moving a model to CPU RAM instead of unloading it.

The 24 GB profile (M5) unloads the phase-A models after phase A, so the next video pays the
full load again (from the network volume). Moving the weights to CPU memory and back should
be much faster. Measured per model, in its own env, on an idle GPU:
- load from disk (what an unloaded model costs on the next job);
- offload to CPU, then restore to the GPU;
- the GPU memory each step frees or takes back.

Pod (workers idle):  envs/asr/bin/python experiments/m6_offload.py whisper
                     envs/sep/bin/python experiments/m6_offload.py roformer
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def gpu_mib() -> int:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.splitlines()[0])


def whisper() -> dict:
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        "Systran/faster-whisper-large-v3",
        revision="edaa852ec7e145841d8ffdb056a99866b5f0a478",
        cache_dir=os.environ.get("DUB_HF_CACHE", "/workspace/dub/hf/hub"),
    )
    m0 = gpu_mib()
    t = time.perf_counter()
    model = WhisperModel(path, device="cuda", compute_type="float16")
    load = time.perf_counter() - t
    m1 = gpu_mib()
    t = time.perf_counter()
    model.model.unload_model(to_cpu=True)
    off = time.perf_counter() - t
    m2 = gpu_mib()
    t = time.perf_counter()
    model.model.load_model()
    on = time.perf_counter() - t
    m3 = gpu_mib()
    return {
        "model": "whisper_large_v3 (CTranslate2)",
        "load_from_disk_s": round(load, 2),
        "offload_s": round(off, 2),
        "restore_s": round(on, 2),
        "gpu_mib": {"loaded": m1 - m0, "freed_by_offload": m1 - m2, "restored": m3 - m2},
    }


def roformer() -> dict:
    import torch
    from audio_separator.separator import Separator

    model_dir = Path(os.environ.get("DUB_MODELS", "/workspace/dub/models")) / "audio-separator"
    m0 = gpu_mib()
    t = time.perf_counter()
    sep = Separator(model_file_dir=str(model_dir), output_dir="/tmp")
    sep.load_model(model_filename="vocals_mel_band_roformer.ckpt")
    torch.cuda.synchronize()
    load = time.perf_counter() - t
    m1 = gpu_mib()
    net = sep.model_instance.model_run
    t = time.perf_counter()
    net.to("cpu")
    torch.cuda.empty_cache()
    off = time.perf_counter() - t
    m2 = gpu_mib()
    t = time.perf_counter()
    net.to("cuda")
    torch.cuda.synchronize()
    on = time.perf_counter() - t
    m3 = gpu_mib()
    return {
        "model": "mel_roformer (torch)",
        "load_from_disk_s": round(load, 2),
        "offload_s": round(off, 2),
        "restore_s": round(on, 2),
        "gpu_mib": {"loaded": m1 - m0, "freed_by_offload": m1 - m2, "restored": m3 - m2},
    }


def main() -> None:
    which = sys.argv[1]
    res = {"whisper": whisper, "roformer": roformer}[which]()
    print(json.dumps(res))
    out = Path(os.environ.get("DUB_EXP_DATA", "/workspace/dub/experiments")) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"m6_offload_{which}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
