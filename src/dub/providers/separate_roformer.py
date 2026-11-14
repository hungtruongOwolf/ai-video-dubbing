"""Speech / background separation with a Mel-Band RoFormer model (python-audio-separator).

Input: 48 kHz stereo WAV. Output: two 48 kHz stereo WAVs in the CAS:
- vocals: the speech (used for ASR on music-heavy content, voice references, diarization);
- instrumental: the "bed" (music + effects) that the dubbed voice is mixed over.

Runs in the `sep` worker env on the GPU pod.

Manifest params:
    model: audio-separator model filename (default: Kim's vocals Mel-Band RoFormer)
    sample_rate: rate the mix is loaded at (default 44100, the model's training rate)
    precision: fp32 | autocast (default) | fp16
    overlap: chunk overlap (default: the model's)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from dub.contracts.v1 import SeparateReq, Stems
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.media import run_ffmpeg

DEFAULT_MODEL = "vocals_mel_band_roformer.ckpt"


class MelRoformer:
    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self.sep: Any = None
        self.loaded: tuple = ()

    def load(self, ctx: RunCtx) -> None:
        from audio_separator.separator import Separator

        model_dir = os.environ.get("DUB_MODELS", str(ctx.cas.root.parent / "models"))
        p = ctx.params
        precision = p.get("precision", "autocast")  # fp32 | autocast | fp16 (native)
        mdxc = {"segment_size": 256, "override_model_segment_size": False,
                "batch_size": None, "overlap": p.get("overlap"), "pitch_shift": 0}  # fmt: skip
        self.sep = Separator(
            output_dir=str(ctx.cas.root / "tmp"),
            model_file_dir=str(Path(model_dir) / "audio-separator"),
            output_format="WAV",
            # The rate the mix is loaded (resampled) at. The model was trained at 44.1 kHz;
            # run() resamples the stems to 48 kHz anyway.
            sample_rate=int(p.get("sample_rate", 44100)),
            mdxc_params=mdxc,
            use_autocast=precision == "autocast",
            use_native_fp16=precision == "fp16",
        )
        self.sep.load_model(model_filename=p.get("model", DEFAULT_MODEL))
        self.loaded = self._key(p)

    @staticmethod
    def _key(p: dict) -> tuple:
        return (p.get("model", DEFAULT_MODEL), p.get("sample_rate", 44100),
                p.get("precision", "autocast"), p.get("overlap"))  # fmt: skip

    def offload(self) -> None:
        """Weights to CPU memory (1.9 GB freed in 0.6 s; back in 0.1 s, against 14.9 s to load)."""
        net = getattr(getattr(self.sep, "model_instance", None), "model_run", None)
        if net is not None:
            import torch

            net.to("cpu")
            torch.cuda.empty_cache()

    def restore(self) -> None:
        net = getattr(getattr(self.sep, "model_instance", None), "model_run", None)
        if net is not None:
            net.to("cuda")

    def run(self, op: str, req: SeparateReq, ctx: RunCtx) -> Stems:
        if op != "audio.separate":
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        if self.sep is None or self.loaded != self._key(ctx.params):
            self.load(ctx)
        out_dir = Path(tempfile.mkdtemp(prefix="sep-", dir=ctx.cas.root / "tmp"))
        try:
            self.sep.output_dir = str(out_dir)
            if getattr(self.sep, "model_instance", None) is not None:
                self.sep.model_instance.output_dir = str(out_dir)
            files = [
                out_dir / Path(f).name for f in self.sep.separate(str(ctx.cas.path(req.audio_key)))
            ]
            vocals = next((f for f in files if "(vocals)" in f.name.lower()), None)
            other = next((f for f in files if f is not vocals), None)
            if vocals is None or other is None:
                raise ProviderError(f"unexpected separator outputs: {[f.name for f in files]}")
            keys = []
            # Normalise both stems to 48 kHz stereo 16-bit for the rest of the pipeline.
            for f in (vocals, other):
                wav = ctx.cas.tmp_path(".wav")
                run_ffmpeg(
                    ["-i", str(f), "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(wav)]
                )
                keys.append(ctx.cas.put_file(wav, move=True))
            return Stems(vocals_key=keys[0], instrumental_key=keys[1])
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
