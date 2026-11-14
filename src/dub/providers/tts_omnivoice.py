"""Zero-shot voice-cloning TTS with OmniVoice (k2-fsa), with exact output duration.

OmniVoice accepts `duration=` in seconds, so the timing ladder can ask for "this line in exactly
2.4 s" (within safe_ratio) instead of stretching audio afterwards. It covers 600+ languages.

The voice prompt (reference clip + its transcript) is encoded once per speaker and cached.
Output: 24 kHz mono WAV in the CAS. Runs in the `tts` worker env on the GPU pod.

Batches (M5): the worker hands queued lines to `run_batch`, which runs them in one
`generate` call (lists of texts, prompts and durations). Sampling stays per line: OmniVoice
draws all its randomness in `_gumbel_sample`, once or twice per line per step; here each line
draws from its own generator, seeded with its own seed, so a line sounds the same whatever it
is batched with (up to float rounding) and the same as when it runs alone.

Manifest params:
    model / revision: Hugging Face repo and commit
    num_step: diffusion steps (32 = quality, 16 = fast drafts)
"""

from __future__ import annotations

import inspect
import threading
from typing import Any

from dub.contracts.v1 import SynthReq, SynthRes
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.audio_io import write_wav

SR = 24000
_CURRENT = threading.local()  # the generator of the line being sampled (see _install_sampler)


def _install_sampler() -> None:
    """Route OmniVoice's random draws to the current line's generator.

    `_generate_iterative` loops over the lines of a batch; for line i it calls
    `_predict_tokens_with_scoring` on a view of row i of the batch logits, then draws the
    position noise. Wrapping that method tells which line is being sampled (from the view's
    storage offset), and the patched `_gumbel_sample` then draws from that line's generator.
    """
    import torch
    from omnivoice.models import omnivoice as ov

    if getattr(ov, "_dub_patched", False):
        return

    def gumbel(logits, temperature):
        scaled = logits / temperature
        gen = getattr(_CURRENT, "gens", None)
        if gen is None:
            u = torch.rand_like(scaled)
        else:
            g = gen[_CURRENT.row]
            u = torch.rand(scaled.shape, generator=g, device=g.device, dtype=scaled.dtype)
        return scaled + -torch.log(-torch.log(u + 1e-10) + 1e-10)

    predict = ov.OmniVoice._predict_tokens_with_scoring

    def predict_with_row(self, c_logits, u_logits, gen_config):
        if getattr(_CURRENT, "gens", None) is not None:
            _CURRENT.row = c_logits.storage_offset() // c_logits.stride(0)
        return predict(self, c_logits, u_logits, gen_config)

    ov._gumbel_sample = gumbel
    ov.OmniVoice._predict_tokens_with_scoring = predict_with_row
    ov._dub_patched = True


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
        _install_sampler()
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
        return self.run_batch(op, [req], ctx)[0]

    def run_batch(self, op: str, reqs: list[SynthReq], ctx: RunCtx) -> list[SynthRes]:
        if op != "tts.synthesize":
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        if self.model is None:
            self.load(ctx)
        import numpy as np
        import torch

        p = ctx.params
        prompts = [self._prompt(ctx, r.voice_key, r.voice_text) for r in reqs]
        kwargs: dict[str, Any] = {
            "text": [r.text for r in reqs],
            "num_step": int(p.get("num_step", 32)),
            "postprocess_output": False,  # keep the exact length we asked for
            "pad_duration": 0.0,
            # OmniVoice fades 100 ms in and out of every take even without post-processing:
            # on a 0.4-1.3 s line that eats the first and last phonemes (M8). The mix fades
            # each clip by 8 ms anyway.
            "fade_duration": float(p.get("fade_duration", 0.01)),
        }
        if any(pr is not None for pr in prompts):
            if any(pr is None for pr in prompts):  # mixed: run the two groups separately
                with_p = [i for i, pr in enumerate(prompts) if pr is not None]
                without = [i for i, pr in enumerate(prompts) if pr is None]
                out: list[SynthRes | None] = [None] * len(reqs)
                for group in (with_p, without):
                    for i, res in zip(group, self.run_batch(op, [reqs[i] for i in group], ctx),
                                      strict=True):  # fmt: skip
                        out[i] = res
                return [r for r in out if r is not None]
            kwargs["voice_clone_prompt"] = prompts
        if any(r.target_ms for r in reqs):
            kwargs["duration"] = [r.target_ms / 1000 if r.target_ms else None for r in reqs]
        if "language" in self.accepts:
            kwargs["language"] = [r.lang for r in reqs]
        dev = self.model.device
        _CURRENT.gens = [torch.Generator(device=dev).manual_seed(int(r.seed)) for r in reqs]
        _CURRENT.row = 0
        try:
            audios = self.model.generate(**kwargs)
        finally:
            _CURRENT.gens = None
        results = []
        for r, a in zip(reqs, audios, strict=True):
            audio = np.asarray(a, dtype=np.float32)
            tmp = ctx.cas.tmp_path(".wav")
            write_wav(tmp, audio, SR)
            results.append(SynthRes(
                unit_id=r.unit_id,
                audio_key=ctx.cas.put_file(tmp, move=True),
                sample_rate=SR,
                dur_ms=round(len(audio) * 1000 / SR),
                provider=self.manifest.id,
                exact=r.target_ms is not None,
            ))  # fmt: skip
        return results
