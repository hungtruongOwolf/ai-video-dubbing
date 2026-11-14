"""Zero-shot voice-cloning TTS with OmniVoice (k2-fsa), with exact output duration.

OmniVoice accepts `duration=` in seconds, so the timing ladder can ask for "this line in exactly
2.4 s" (within safe_ratio) instead of stretching audio afterwards. It covers 600+ languages.

The voice prompt (reference clip + its transcript) is encoded once per speaker and cached.
Output: 24 kHz mono WAV in the CAS. Runs in the `tts` worker env on the GPU pod.

Manifest params:
    model / revision: Hugging Face repo and commit
    num_step: diffusion steps (32 = quality, 16 = fast drafts)
"""

from __future__ import annotations

import inspect
from typing import Any

from dub.contracts.v1 import SynthReq, SynthRes
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.audio_io import write_wav

SR = 24000


class OmniVoiceTTS:
    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self.model: Any = None
        self.prompts: dict[tuple[str, str | None], Any] = {}
        self.accepts: set[str] = set()

    def load(self, ctx: RunCtx) -> None:
        import torch
        from omnivoice import OmniVoice

        p = ctx.params
        kwargs: dict[str, Any] = {"device_map": "cuda:0", "dtype": torch.float16}
        if p.get("revision"):
            kwargs["revision"] = p["revision"]
        self.model = OmniVoice.from_pretrained(p.get("model", "k2-fsa/OmniVoice"), **kwargs)
        self.accepts = set(inspect.signature(self.model.generate).parameters)
        # One tiny generation so CUDA kernels are compiled and cached now, not on the first
        # line a user edits (that first call was ~3 s slower).
        try:
            self.model.generate(text="Hello.", num_step=4, duration=0.5)
        except Exception as e:  # noqa: BLE001 - warm-up is best effort
            ctx.log(f"warm-up skipped: {e}")

    def _prompt(self, ctx: RunCtx, voice_key: str | None, voice_text: str | None) -> Any:
        if not voice_key:
            return None
        k = (voice_key, voice_text)
        if k not in self.prompts:
            self.prompts[k] = self.model.create_voice_clone_prompt(
                ref_audio=str(ctx.cas.path(voice_key)), ref_text=voice_text
            )
        return self.prompts[k]

    def run(self, op: str, req: SynthReq, ctx: RunCtx) -> SynthRes:
        if op != "tts.synthesize":
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        if self.model is None:
            self.load(ctx)
        import numpy as np
        import torch

        p = ctx.params
        kwargs: dict[str, Any] = {
            "text": req.text,
            "num_step": int(p.get("num_step", 32)),
            "postprocess_output": False,  # keep the exact length we asked for
            "pad_duration": 0.0,
        }
        prompt = self._prompt(ctx, req.voice_key, req.voice_text)
        if prompt is not None:
            kwargs["voice_clone_prompt"] = prompt
        if req.target_ms:
            kwargs["duration"] = req.target_ms / 1000
        if "language" in self.accepts:
            kwargs["language"] = req.lang
        torch.manual_seed(req.seed)
        audio = np.asarray(self.model.generate(**kwargs)[0], dtype=np.float32)
        tmp = ctx.cas.tmp_path(".wav")
        write_wav(tmp, audio, SR)
        return SynthRes(
            unit_id=req.unit_id,
            audio_key=ctx.cas.put_file(tmp, move=True),
            sample_rate=SR,
            dur_ms=round(len(audio) * 1000 / SR),
            provider=self.manifest.id,
            exact=req.target_ms is not None,
        )
