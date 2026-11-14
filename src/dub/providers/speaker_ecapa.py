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


MAX_EMBED_S = 20  # longer chunks are embedded on their middle 20 s
BATCH_SAMPLES = 64 * 3 * 16000  # padded samples per batch (64 windows of 3 s)


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
        """L2-normalised embeddings, in the order of `chunks`.

        Batches are built by length (shortest first) under a budget of padded samples, and a
        chunk longer than MAX_EMBED_S is embedded on its middle part: on a 60-minute speech
        the pause-delimited pieces reach minutes, and 64 of them padded to the longest ran a
        24 GB GPU out of memory.
        """
        import torch

        if not chunks:
            return np.zeros((0, 192), dtype=np.float32)
        cap = MAX_EMBED_S * SR
        prepared = []
        for c in chunks:
            c = np.asarray(c, dtype=np.float32)
            if len(c) > cap:
                mid = (len(c) - cap) // 2
                c = c[mid : mid + cap]
            if len(c) < SR // 4:
                c = np.pad(c, (0, SR // 4 - len(c)))
            prepared.append(c)
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]))
        out = np.zeros((len(prepared), 192), dtype=np.float32)
        i = 0
        while i < len(order):
            j = i + 1
            while (
                j < len(order)
                and j - i < 64
                and (j + 1 - i) * len(prepared[order[j]]) <= (BATCH_SAMPLES)
            ):
                j += 1
            idx = order[i:j]
            n = len(prepared[idx[-1]])
            x = torch.zeros(len(idx), n)
            for k, t in enumerate(idx):
                x[k, : len(prepared[t])] = torch.from_numpy(prepared[t])
            lens = torch.tensor([len(prepared[t]) / n for t in idx])
            with torch.no_grad():
                e = self.enc.encode_batch(x.to(self.enc.device), lens.to(self.enc.device))
            e = torch.nn.functional.normalize(e.squeeze(1), dim=-1)
            out[idx] = e.float().cpu().numpy()
            i = j
        return out

    def run(self, op: str, req, ctx: RunCtx):
        if self.enc is None:
            self.load(ctx)
        if op == "audio.embed":
            return self._op_embed(req, ctx)
        if op == "audio.diarize":
            return self._op_diarize(req, ctx)
        raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")

    def _resegment(self, mono, embs, labels, wins, coarse, p) -> list:
        """Re-cut the turns at real pauses (see dub.ops.diarize.resegment)."""
        cents = diar_ops.centroids(embs, np.asarray(labels))
        keys = sorted(cents)
        cmat = np.stack([cents[k] for k in keys])
        gap = int(p.get("reseg_gap_ms", 60))
        pieces = energy_vad(
            mono, SR, frame_ms=10, min_speech_ms=100, merge_gap_ms=gap, floor_db=-50.0
        )
        min_ms = int(p.get("reseg_min_ms", 400))
        idx = [i for i, (s, e) in enumerate(pieces) if e - s >= min_ms]
        if not idx:
            return coarse
        vecs = self.embed(
            [mono[pieces[i][0] * SR // 1000 : pieces[i][1] * SR // 1000] for i in idx]
        )
        best = np.argmax(vecs @ cmat.T, axis=1)
        piece_labels = {i: int(keys[b]) for i, b in zip(idx, best, strict=True)}
        # coarse turns carry S1.. names; map them to "#<label>" via majority overlap with pieces
        name_of: dict[str, str] = {}
        for t in coarse:
            votes: dict[int, int] = {}
            for i, lab in piece_labels.items():
                s, e = pieces[i]
                ov = min(e, t.end_ms) - max(s, t.start_ms)
                if ov > 0:
                    votes[lab] = votes.get(lab, 0) + ov
            if votes:
                name_of[t.speaker] = f"#{max(votes, key=votes.get)}"
        coarse_named = [
            diar_ops.Turn(speaker=name_of.get(t.speaker, f"#{t.speaker}"), start_ms=t.start_ms,
                          end_ms=t.end_ms)
            for t in coarse
        ]  # fmt: skip
        return diar_ops.resegment(pieces, piece_labels, coarse_named)

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
            labels = fcluster(z, t=float(p.get("threshold", 0.85)), criterion="distance")
        labels = diar_ops.absorb_small(
            embs,
            np.asarray(labels),
            wins,
            min_share=float(p.get("min_share", 0.05)),
            min_windows=int(p.get("min_windows", 3)),
        )
        turns = diar_ops.turns_from_windows(wins, labels, regions)
        if p.get("reseg", True):
            turns = self._resegment(mono, embs, labels, wins, turns, p)
        speakers = sorted({t.speaker for t in turns}, key=lambda s: int(s[1:]))
        return Diarization(turns=turns, speakers=speakers)
