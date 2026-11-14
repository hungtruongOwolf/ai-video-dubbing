"""Speaker embeddings (ECAPA-TDNN, SpeechBrain, VoxCeleb) and diarization by clustering them.

Two ops share the model:
- audio.embed: one 192-d unit vector per audio span. Used to pick the cleanest voice
  reference of each speaker and, in the QA gate, to check that a dubbed line still sounds
  like its speaker (cosine similarity).
- audio.diarize: who speaks when. VAD on the separated vocals -> 1.5 s windows -> embeddings
  -> agglomerative clustering (average linkage, cosine distance; a fixed number of speakers
  when the job gives `speakers_hint`) -> tiny clusters folded into real ones -> turns.
  Pyannote's pipelines are better at overlapping speech but gated behind a Hugging Face
  account; this one needs no token. Experiment E8 measures it.

Runs in the `tts` worker env (torch). Apache-2.0 model.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from dub.contracts.v1 import (
    Diarization,
    DiarizeReq,
    Embeddings,
    EmbedReq,
)
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops import diarize as diar_ops
from dub.ops.audio_io import read_wav, resample_int
from dub.ops.vad import energy_vad

SR = 16000


def _mono16(path) -> np.ndarray:
    audio, sr = read_wav(path)
    mono = audio.mean(axis=1)
    if sr == SR:
        return mono.astype(np.float32)
    if sr % SR == 0:
        return resample_int(mono, 1, sr // SR)
    if (2 * sr) % SR == 0:
        return resample_int(mono, 2, 2 * sr // SR)
    raise ProviderError(f"unsupported sample rate {sr}", kind="bad_input")


class EcapaSpeaker:
    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self.enc: Any = None

    def load(self, ctx: RunCtx) -> None:
        from huggingface_hub import snapshot_download
        from speechbrain.inference.speaker import EncoderClassifier

        p = ctx.params
        path = snapshot_download(
            p["model"], revision=p.get("revision"), cache_dir=os.environ.get("DUB_HF_CACHE")
        )
        save = os.path.join(os.environ.get("DUB_MODELS", "/tmp"), "ecapa", p["revision"][:12])
        self.enc = EncoderClassifier.from_hparams(
            source=path, savedir=save, run_opts={"device": p.get("device", "cuda")}
        )

    def embed(self, chunks: list[np.ndarray]) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(chunks), 64):
            batch = [
                c if len(c) >= SR // 4 else np.pad(c, (0, SR // 4 - len(c)))
                for c in chunks[i : i + 64]
            ]
            n = max(len(c) for c in batch)
            x = torch.zeros(len(batch), n)
            for j, c in enumerate(batch):
                x[j, : len(c)] = torch.from_numpy(np.asarray(c, dtype=np.float32))
            lens = torch.tensor([len(c) / n for c in batch])
            with torch.no_grad():
                e = self.enc.encode_batch(x.to(self.enc.device), lens.to(self.enc.device))
            e = torch.nn.functional.normalize(e.squeeze(1), dim=-1)
            out.append(e.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 192), dtype=np.float32)

    def run(self, op: str, req, ctx: RunCtx):
        if self.enc is None:
            self.load(ctx)
        if op == "audio.embed":
            return self._op_embed(req, ctx)
        if op == "audio.diarize":
            return self._op_diarize(req, ctx)
        raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")

    def _op_embed(self, req: EmbedReq, ctx: RunCtx) -> Embeddings:
        mono = _mono16(ctx.cas.path(req.audio_key))
        spans = req.spans or [(0, round(len(mono) * 1000 / SR))]
        chunks = [mono[s * SR // 1000 : max(e * SR // 1000, s * SR // 1000 + 1)] for s, e in spans]
        vecs = self.embed(chunks)
        return Embeddings(vectors=[[round(float(v), 4) for v in row] for row in vecs])

    def _op_diarize(self, req: DiarizeReq, ctx: RunCtx) -> Diarization:
        from scipy.cluster.hierarchy import fcluster, linkage

        p = ctx.params
        mono = _mono16(ctx.cas.path(req.audio_key))
        regions = energy_vad(mono, SR, frame_ms=10, min_speech_ms=120, merge_gap_ms=150,
                             floor_db=-50.0)  # fmt: skip
        wins = diar_ops.windows(regions, int(p.get("win_ms", 1500)), int(p.get("hop_ms", 750)))
        if len(wins) < 2:
            end = round(len(mono) * 1000 / SR)
            return Diarization(turns=[diar_ops.Turn(speaker="S1", start_ms=0, end_ms=end)],
                               speakers=["S1"])  # fmt: skip
        embs = self.embed([mono[s * SR // 1000 : e * SR // 1000] for s, e in wins])
        z = linkage(embs, method="average", metric="cosine")
        if req.speakers_hint:
            labels = fcluster(z, t=req.speakers_hint, criterion="maxclust")
        else:
            labels = fcluster(z, t=float(p.get("threshold", 0.6)), criterion="distance")
        labels = diar_ops.absorb_small(
            embs,
            np.asarray(labels),
            wins,
            min_share=float(p.get("min_share", 0.02)),
            min_windows=int(p.get("min_windows", 3)),
        )
        turns = diar_ops.turns_from_windows(wins, labels, regions)
        speakers = sorted({t.speaker for t in turns}, key=lambda s: int(s[1:]))
        return Diarization(turns=turns, speakers=speakers)
