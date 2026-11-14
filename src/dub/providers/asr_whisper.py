"""Speech-to-text with faster-whisper (CTranslate2), with word timestamps.

Used for Whisper large-v3 and for PhoWhisper (a Vietnamese fine-tune of Whisper converted to
CTranslate2). Runs in the `asr` worker env on the GPU pod.

Two ops share the model:
- asr.transcribe: the original audio (source language);
- asr.backcheck: one dubbed clip (target language). Its words time the subtitles of the
  dub and give the line's CER. It never sees the expected text.

Batches (M5): queued back-checks are concatenated with short silences and decoded together
by faster-whisper's BatchedInferencePipeline, one clip per `clip_timestamps` entry; the words
are then split back by time.

Manifest params:
    model:        Hugging Face repo id of a CTranslate2 model
    revision:     commit to pin (weights_rev in the cache key)
    language:     force a language (PhoWhisper: "vi"); otherwise the request's language or
                  Whisper's own detection
    compute_type: float16 on GPU
    beam_size:    5
"""

from __future__ import annotations

import os
from typing import Any

from dub.contracts.v1 import TranscribeReq, Word, WordTimeline
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.audio_io import read_wav

GAP_S = 0.5  # silence between the clips of a batched back-check
MAX_BATCH_S = 28.0  # a clip longer than this is checked on its own (Whisper's window is 30 s)


class FasterWhisperASR:
    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self.model: Any = None
        self.batched: Any = None

    def load(self, ctx: RunCtx) -> None:
        from faster_whisper import WhisperModel
        from huggingface_hub import snapshot_download

        p = ctx.params
        path = snapshot_download(
            p["model"], revision=p.get("revision"), cache_dir=os.environ.get("DUB_HF_CACHE")
        )
        self.model = WhisperModel(
            path, device=p.get("device", "cuda"), compute_type=p.get("compute_type", "float16")
        )

    def run(self, op: str, req: TranscribeReq, ctx: RunCtx) -> WordTimeline:
        if op not in ("asr.transcribe", "asr.backcheck"):
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        if self.model is None:
            self.load(ctx)
        p = ctx.params
        forced = p.get("language")
        lang = forced or (None if req.lang in ("", "auto") else req.lang)
        # Pass decoded samples: our input is already a 16 kHz mono WAV, and this avoids
        # faster-whisper's PyAV decoder (newer PyAV releases broke its call signature).
        audio, sr = read_wav(ctx.cas.path(req.audio_key))
        if sr != 16000:
            raise ProviderError(f"expected 16 kHz audio, got {sr} Hz", kind="bad_input")
        extra: dict = {}
        if req.clips:  # one speaker turn per clip: Whisper cannot merge words across turns
            extra["clip_timestamps"] = [round(t / 1000, 3) for c in req.clips for t in c]
        segments, info = self.model.transcribe(
            audio.mean(axis=1),
            language=lang,
            word_timestamps=True,
            vad_filter=bool(p.get("vad_filter", True)) and not req.clips,
            beam_size=int(p.get("beam_size", 5)),
            condition_on_previous_text=False,  # fewer runaway repetitions on long files
            **extra,
        )
        words: list[Word] = []
        for seg in segments:  # a generator: decoding happens while we iterate
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append(
                        Word(
                            w=text,
                            start_ms=round(w.start * 1000),
                            end_ms=round(w.end * 1000),
                            conf=round(float(w.probability), 3),
                            # zh/ja/th "words" are pieces; keep the spaces the model wrote
                            space_before=bool(words) and w.word[:1].isspace(),
                        )
                    )
        return WordTimeline(lang=lang or info.language, words=words)

    def run_batch(self, op: str, reqs: list[TranscribeReq], ctx: RunCtx) -> list[WordTimeline]:
        if op != "asr.backcheck" or len(reqs) < 2:
            return [self.run(op, r, ctx) for r in reqs]
        if self.model is None:
            self.load(ctx)
        import numpy as np
        from faster_whisper import BatchedInferencePipeline

        if self.batched is None:
            self.batched = BatchedInferencePipeline(self.model)
        p = ctx.params
        out: list[WordTimeline | None] = [None] * len(reqs)
        groups: dict[str | None, list[int]] = {}
        clips: dict[int, np.ndarray] = {}
        for i, r in enumerate(reqs):
            audio, sr = read_wav(ctx.cas.path(r.audio_key))
            mono = audio.mean(axis=1).astype(np.float32)
            if sr != 16000 or len(mono) / 16000 > MAX_BATCH_S or r.clips:
                out[i] = self.run(op, r, ctx)
                continue
            clips[i] = mono
            lang = p.get("language") or (None if r.lang in ("", "auto") else r.lang)
            groups.setdefault(lang, []).append(i)
        gap = np.zeros(int(GAP_S * 16000), dtype=np.float32)
        for lang, idx in groups.items():
            parts, spans, t = [], [], 0.0
            for i in idx:
                d = len(clips[i]) / 16000
                parts += [clips[i], gap]
                spans.append((t, t + d))
                t += d + GAP_S
            segments, info = self.batched.transcribe(
                np.concatenate(parts),
                language=lang,
                word_timestamps=True,
                clip_timestamps=[{"start": s0, "end": s1} for s0, s1 in spans],
                batch_size=min(16, len(idx)),
                beam_size=int(p.get("beam_size", 5)),
            )
            per: list[list[Word]] = [[] for _ in idx]
            for seg in segments:
                for w in seg.words or []:
                    text = w.word.strip()
                    if not text:
                        continue
                    k = next((j for j, (s0, s1) in enumerate(spans)
                              if w.start < s1 + GAP_S / 2), len(spans) - 1)  # fmt: skip
                    s0 = spans[k][0]
                    words = per[k]
                    words.append(Word(
                        w=text,
                        start_ms=max(0, round((w.start - s0) * 1000)),
                        end_ms=max(0, round((w.end - s0) * 1000)),
                        conf=round(float(w.probability), 3),
                        space_before=bool(words) and w.word[:1].isspace(),
                    ))  # fmt: skip
            for j, i in enumerate(idx):
                out[i] = WordTimeline(lang=lang or info.language, words=per[j])
        return [o for o in out if o is not None]
