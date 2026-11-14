"""Sentence-boundary probabilities with SaT ("Segment any Text", wtpsplit), 85 languages.

SaT reads raw text and gives, for every character, the probability that a sentence ends
right after it. It was trained to cope with text that has no punctuation and no casing,
which is exactly what many ASR models produce (PhoWhisper, Whisper on Thai...). The punctuation
rules cannot do anything with such text.

Words -> one text (spaces between words, or the ASR's own spaces for zh/ja/th), one
probability per word: the probability at its last character, or at the space after it.

Calibration: the manifest's `threshold` (chosen by experiment E4) is mapped to 0.5, so the
segmenter can use one threshold for every provider:  p' = sigmoid(logit(p) - logit(t)).

Runs in the `text` worker env on the GPU pod (fp16 on the GPU: ~0.6 GB of VRAM).
"""

from __future__ import annotations

import math
import os
from typing import Any

from dub.contracts.v1 import Boundaries, BoundaryReq
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx


def build_text(req: BoundaryReq) -> tuple[str, list[int]]:
    """The text SaT reads, and the index just past each word in it."""
    parts: list[str] = []
    ends: list[int] = []
    pos = 0
    for i, w in enumerate(req.words):
        w = w.strip()
        sep = ""
        if i > 0:
            if req.joiner:
                sep = req.joiner
            elif i < len(req.space_before) and req.space_before[i]:
                sep = " "
        parts.append(sep + w)
        pos += len(sep) + len(w)
        ends.append(pos)
    return "".join(parts), ends


def word_probs(char_probs: Any, text: str, ends: list[int]) -> list[float]:
    out = []
    for e in ends:
        p = float(char_probs[e - 1]) if e > 0 else 0.0
        if e < len(text) and text[e].isspace():
            p = max(p, float(char_probs[e]))
        out.append(p)
    return out


def calibrate(p: float, threshold: float) -> float:
    """Map the model's decision threshold to 0.5, keeping the order of probabilities."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = math.log(p / (1 - p)) - math.log(threshold / (1 - threshold))
    return round(1 / (1 + math.exp(-z)), 4)


class SaTBoundaries:
    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self.sat: Any = None

    def load(self, ctx: RunCtx) -> None:
        from huggingface_hub import snapshot_download
        from wtpsplit import SaT

        p = ctx.params
        path = snapshot_download(
            p["model"],
            revision=p.get("revision"),
            cache_dir=os.environ.get("DUB_HF_CACHE"),
            allow_patterns=["config.json", "model.safetensors"],  # skip the ONNX copies
        )
        self.sat = SaT(path)
        if p.get("device", "cuda") == "cuda":
            self.sat.half().to("cuda")

    def run(self, op: str, req: BoundaryReq, ctx: RunCtx) -> Boundaries:
        if op != "text.boundaries":
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        if not req.words:
            return Boundaries(probs=[])
        if self.sat is None:
            self.load(ctx)
        text, ends = build_text(req)
        char_probs = self.sat.predict_proba(text)
        t = float(ctx.params.get("threshold", 0.25))
        return Boundaries(probs=[calibrate(p, t) for p in word_probs(char_probs, text, ends)])
