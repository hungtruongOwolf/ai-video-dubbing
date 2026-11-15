"""Vietnamese TTS with VieNeu-TTS v3 Turbo (pnnbao-ump, Apache-2.0): voice cloning, 48 kHz.

A second engine next to OmniVoice, with the other kind of length control: VieNeu is
autoregressive and has no `duration`, so the timing ladder treats it as "none" (pick the
variant that fits, borrow silence, stretch the rest). Its GPU engine batches the chunks of
a line in one forward pass and captures each frame step as a CUDA graph.

The reference clip of each speaker is enrolled once (`add_voice`: denoise + speaker profile)
and reused by name. Seeds: the sampler draws from torch's RNG, seeded per request.

Runs in the `tts2` worker env (its GPU build pins transformers 4.57.6, which OmniVoice's env
cannot share).
"""

from __future__ import annotations

from typing import Any

from dub.contracts.v1 import SynthReq, SynthRes
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.audio_io import write_wav

SR = 48000


class VieNeuTTS:
    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self.tts: Any = None
        self.voices: set[str] = set()

    def load(self, ctx: RunCtx) -> None:
        from vieneu import Vieneu

        p = ctx.params
        self.tts = Vieneu(
            mode="v3turbo", backbone_repo=p.get("model", "pnnbao-ump/VieNeu-TTS-v3-Turbo")
        )
        # Warm-up: every path a real line takes, now and not on the first line of a job (the
        # graphs, the watermark, and enrolling a voice, which loads the denoiser). Loaded
        # lazily, the first lines of a job waited 130-196 s for it.
        import numpy as np

        watermark = bool(p.get("watermark", True))
        try:
            audio = self.tts.infer("Xin chào, đây là câu thử giọng.", apply_watermark=watermark)
            tmp = ctx.cas.tmp_path(".wav")
            write_wav(tmp, np.asarray(audio, dtype=np.float32), SR)
            self.tts.add_voice("_warmup", str(tmp), denoise=True)
            self.tts.infer("Một hai ba.", voice="_warmup", apply_watermark=watermark)
            tmp.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001 - warm-up is best effort
            ctx.log(f"warm-up incomplete: {e}")

    def _voice(self, ctx: RunCtx, voice_key: str | None) -> str | None:
        if not voice_key:
            return None
        if voice_key not in self.voices:
            self.tts.add_voice(voice_key, str(ctx.cas.path(voice_key)), denoise=True)
            self.voices.add(voice_key)
        return voice_key

    def run(self, op: str, req: SynthReq, ctx: RunCtx) -> SynthRes:
        if op != "tts.synthesize":
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        if self.tts is None:
            self.load(ctx)
        import numpy as np
        import torch

        torch.manual_seed(req.seed)
        p = ctx.params
        audio = self.tts.infer(
            req.text,
            voice=self._voice(ctx, req.voice_key),
            temperature=float(p.get("temperature", 0.8)),
            apply_watermark=bool(p.get("watermark", True)),
        )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ProviderError("empty audio", kind="bad_output", retryable=True)
        tmp = ctx.cas.tmp_path(".wav")
        write_wav(tmp, audio, SR)
        return SynthRes(
            unit_id=req.unit_id,
            audio_key=ctx.cas.put_file(tmp, move=True),
            sample_rate=SR,
            dur_ms=round(len(audio) * 1000 / SR),
            provider=self.manifest.id,
            exact=False,
        )
