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
WARM_BATCH = 8  # = limits.max_batch in the manifest


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
            # the batch sizes a job sends (lines queued at the worker run as one batch)
            for n in (2, 4, WARM_BATCH):
                self.tts.infer_batch(["Một hai ba bốn."] * n, voice="_warmup",
                                     apply_watermark=watermark)  # fmt: skip
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
        import torch

        torch.manual_seed(req.seed)
        p = ctx.params
        audio = self.tts.infer(
            req.text,
            voice=self._voice(ctx, req.voice_key),
            temperature=float(p.get("temperature", 0.8)),
            apply_watermark=bool(p.get("watermark", True)),
        )
        return self._result(req, audio, ctx)

    def run_batch(self, op: str, reqs: list[SynthReq], ctx: RunCtx) -> list[SynthRes]:
        """Lines queued at the worker, one forward pass per voice (`infer_batch`: every
        chunk of every line shares the decoding steps). One at a time, the drama's 151 short
        lines took 105 s in series while 10-21 waited. Sampling runs batched, so a take is
        equivalent but not bit-identical to the same line voiced alone (as with OmniVoice,
        M5); the seed of a group is that of its first line."""
        if op != "tts.synthesize" or len(reqs) < 2:
            return [self.run(op, r, ctx) for r in reqs]
        if self.tts is None:
            self.load(ctx)
        import torch

        p = ctx.params
        groups: dict[str | None, list[int]] = {}
        for i, r in enumerate(reqs):
            groups.setdefault(r.voice_key, []).append(i)
        out: list[SynthRes | None] = [None] * len(reqs)
        for key, idx in groups.items():
            torch.manual_seed(reqs[idx[0]].seed)
            wavs = self.tts.infer_batch(
                [reqs[i].text for i in idx],
                voice=self._voice(ctx, key),
                temperature=float(p.get("temperature", 0.8)),
                apply_watermark=bool(p.get("watermark", True)),
            )
            for i, w in zip(idx, wavs, strict=True):
                out[i] = self._result(reqs[i], w, ctx)
        return out  # type: ignore[return-value]

    def _result(self, req: SynthReq, audio, ctx: RunCtx) -> SynthRes:
        import numpy as np

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
