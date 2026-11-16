"""Emotion embeddings of speech: emotion2vec+ large (M13).

emotion2vec (Ma et al. 2024, arXiv 2312.15185) is a self-supervised speech model, fine-tuned
in emotion2vec+ on pseudo-labelled emotional speech in many languages. Each clip gets:
- an utterance embedding: the mean of its frame features, which places a line by how it is
  said (anger and joy at the same pitch differ here, unlike in dub.ops.prosody);
- the probabilities of 9 classes: angry, disgusted, fearful, happy, neutral, other, sad,
  surprised, unknown.

Weights: emotion2vec/emotion2vec_plus_large on Hugging Face (~300M parameters), under the
FunASR Model Open Source License 1.1. It allows use, copying, modification and sharing, with
the source and model name kept. Run through FunASR (MIT), in its own env (`emo`): FunASR's
dependencies stay away from OmniVoice's and VieNeu's pins. Input: 16 kHz mono; clips are read
from the CAS and resampled when they are 24 or 48 kHz.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from dub.contracts.v1 import EmotionReq, Emotions
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.audio_io import read_wav, resample_int

SR = 16000
MIN_S = 0.3  # shorter clips get no embedding
MAX_S = 15  # longer clips are read on their middle part
RATIO = {16000: (1, 1), 24000: (2, 3), 48000: (1, 3)}


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr not in RATIO:
        raise ProviderError(f"emotion2vec reads 16, 24 or 48 kHz audio, got {sr} Hz",
                            kind="bad_input")  # fmt: skip
    up, down = RATIO[sr]
    return audio if up == down else resample_int(audio, up, down)


def english(label: str) -> str:
    """FunASR's labels are "生气/angry": the English half."""
    return label.split("/")[-1].strip().lower()


class Emotion2Vec:
    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self.model: Any = None

    def load(self, ctx: RunCtx) -> None:
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        p = ctx.params
        path = snapshot_download(
            p["model"],
            revision=p.get("revision"),
            cache_dir=os.environ.get("DUB_HF_CACHE"),
            allow_patterns=["config.yaml", "configuration.json", "model.pt", "tokens.txt"],
        )
        self.model = AutoModel(model=path, device=p.get("device", "cuda:0"), disable_update=True,
                               disable_pbar=True, log_level="ERROR")  # fmt: skip

    def run(self, op: str, req: EmotionReq, ctx: RunCtx) -> Emotions:
        if op != "audio.emotion":
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        if self.model is None:
            self.load(ctx)
        files: dict[str, np.ndarray] = {}
        vectors: list[list[float]] = []
        scores: list[list[float]] = []
        labels: list[str] = []
        for key, s, e in req.items:
            if key not in files:
                audio, sr = read_wav(ctx.cas.path(key))
                files[key] = to_16k(audio.mean(axis=1).astype(np.float32), sr)
            mono = files[key]
            x = mono[s * SR // 1000 : (e * SR // 1000 if e else len(mono))]
            if len(x) < MIN_S * SR:
                vectors.append([])
                scores.append([])
                continue
            if len(x) > MAX_S * SR:
                x = x[(len(x) - MAX_S * SR) // 2 :][: MAX_S * SR]
            res = self.model.generate(input=np.ascontiguousarray(x), granularity="utterance",
                                      extract_embedding=True, disable_pbar=True)  # fmt: skip
            r = res[0]
            v = np.asarray(r["feats"], dtype=np.float32).reshape(-1)
            vectors.append([round(float(t), 5) for t in v])
            scores.append([round(float(t), 5) for t in r.get("scores", [])])
            labels = labels or [english(lab) for lab in r.get("labels", [])]
        return Emotions(vectors=vectors, scores=scores, labels=labels)
