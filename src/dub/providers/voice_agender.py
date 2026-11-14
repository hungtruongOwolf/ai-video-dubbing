"""Age and gender from the voice (audeering wav2vec2-large-robust-24-ft-age-gender).

The model (CC BY-NC-SA 4.0: non-commercial use) reads 16 kHz speech and returns an age
(0..1 for 0..100 years) and the probabilities of female, male and child. Each speaker gets the
average over a few clean spans of their own speech (weighted by length), turned into the
`SpeakerVoice` hint that the cast-sheet pass reads as weak evidence: below `min_conf` the
gender stays "unknown". What people say about themselves ("my son", "she") always outranks it.

Runs in the `tts` worker env (torch and transformers are already there), loaded on first use.
Model code from the model card.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from dub.contracts.v1 import SpeakerVoice, VoiceProfileReq, VoiceProfiles
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.audio_io import read_wav

SR = 16000
MAX_SPAN_S = 10  # longer spans are read on their middle part


def age_band(age_years: float, p_child: float) -> str:
    if p_child >= 0.5 or age_years < 13:
        return "child"
    if age_years < 25:
        return "young"
    if age_years < 60:
        return "adult"
    return "senior"


def to_voice(speaker: str, age01: float, p_f: float, p_m: float, p_c: float,
             min_conf: float) -> SpeakerVoice:  # fmt: skip
    """Scores -> hint. Gender from female vs male only (a child still has one)."""
    fm = p_f + p_m
    p_female = p_f / fm if fm > 0 else 0.5
    conf = max(p_female, 1 - p_female)
    gender = ("female" if p_female >= 0.5 else "male") if conf >= min_conf else "unknown"
    return SpeakerVoice(
        speaker=speaker,
        gender=gender,
        gender_conf=round(conf, 3),
        age_band=age_band(age01 * 100, p_c),
    )


class AgeGenderVoice:
    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self.model: Any = None
        self.processor: Any = None

    def load(self, ctx: RunCtx) -> None:
        import torch
        import torch.nn as nn
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class ModelHead(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, x):
                x = torch.tanh(self.dense(self.dropout(x)))
                return self.out_proj(self.dropout(x))

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.post_init()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

        p = ctx.params
        path = snapshot_download(
            p["model"],
            revision=p.get("revision"),
            cache_dir=os.environ.get("DUB_HF_CACHE"),
            allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json",
                            "vocab.json"],
        )  # fmt: skip
        self.processor = Wav2Vec2Processor.from_pretrained(path)
        self.model = AgeGenderModel.from_pretrained(path).eval()
        if p.get("device", "cuda") == "cuda":
            self.model = self.model.to("cuda")

    def offload(self) -> None:
        if self.model is not None:
            import torch

            self.model.to("cpu")
            torch.cuda.empty_cache()

    def restore(self) -> None:
        if self.model is not None:
            self.model.to("cuda")

    def run(self, op: str, req: VoiceProfileReq, ctx: RunCtx) -> VoiceProfiles:
        if op != "audio.voice_profile":
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        if self.model is None:
            self.load(ctx)
        import torch

        audio, sr = read_wav(ctx.cas.path(req.audio_key))
        if sr != SR:
            raise ProviderError(f"expected 16 kHz audio, got {sr} Hz", kind="bad_input")
        mono = audio.mean(axis=1).astype(np.float32)
        dev = next(self.model.parameters()).device
        acc: dict[str, list[tuple[float, np.ndarray]]] = {}
        for (s, e), spk in zip(req.spans, req.speakers, strict=True):
            x = mono[s * SR // 1000 : e * SR // 1000]
            if len(x) < SR // 2:
                continue
            cap = MAX_SPAN_S * SR
            if len(x) > cap:
                x = x[(len(x) - cap) // 2 :][:cap]
            y = self.processor(x, sampling_rate=SR)["input_values"][0].reshape(1, -1)
            with torch.no_grad():
                age, gender = self.model(torch.from_numpy(np.asarray(y)).to(dev))
            scores = np.array([float(age[0, 0]), *[float(v) for v in gender[0]]])
            acc.setdefault(spk, []).append((len(x) / SR, scores))
        min_conf = float(ctx.params.get("min_conf", 0.7))
        voices, detail = [], {}
        for spk, rows in sorted(acc.items()):
            w = np.array([d for d, _ in rows])
            m = (np.stack([r for _, r in rows]) * w[:, None]).sum(axis=0) / w.sum()
            age01, p_f, p_m, p_c = (float(v) for v in m)
            voices.append(to_voice(spk, age01, p_f, p_m, p_c, min_conf))
            detail[spk] = {"age": round(age01 * 100, 1), "female": round(p_f, 3),
                           "male": round(p_m, 3), "child": round(p_c, 3),
                           "seconds": round(float(w.sum()), 1)}  # fmt: skip
        return VoiceProfiles(voices=voices, detail=detail)
