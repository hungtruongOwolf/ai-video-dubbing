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
    batch_size:   0 = sequential decoding of the source; > 0 = batched (VAD chunks)
"""

from __future__ import annotations

import os
import threading
from typing import Any

from dub.contracts.v1 import TranscribeReq, Word, WordTimeline
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.audio_io import read_wav

# One model object per (weights, device, precision) in a worker process: providers that use
# the same weights (whisper_large_v3 and the punctuation donor of phowhisper_punct) share it
# instead of holding two copies on the GPU. The worker runs one call at a time, so sharing is
# safe; a provider that finds the model parked in CPU memory by another one brings it back.
_MODELS: dict[tuple, Any] = {}
_MODELS_LOCK = threading.Lock()

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
        key = (path, p.get("device", "cuda"), p.get("compute_type", "float16"))
        with _MODELS_LOCK:
            if key not in _MODELS:
                _MODELS[key] = WhisperModel(path, device=key[1], compute_type=key[2])
            self.model = _MODELS[key]

    def _ready(self, ctx: RunCtx) -> None:
        """Loaded, and on the GPU (a provider sharing the weights may have parked them)."""
        if self.model is None:
            self.load(ctx)
        if not self.model.model.model_is_loaded:
            self.model.model.load_model()

    def run(self, op: str, req: TranscribeReq, ctx: RunCtx) -> WordTimeline:
        if op not in ("asr.transcribe", "asr.backcheck"):
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        self._ready(ctx)
        p = ctx.params
        forced = p.get("language")
        lang = forced or (None if req.lang in ("", "auto") else req.lang)
        # Pass decoded samples: our input is already a 16 kHz mono WAV, and this avoids
        # faster-whisper's PyAV decoder (newer PyAV releases broke its call signature).
        audio, sr = read_wav(ctx.cas.path(req.audio_key))
        if sr != 16000:
            raise ProviderError(f"expected 16 kHz audio, got {sr} Hz", kind="bad_input")
        extra: dict = {}
        # A punctuated example nudges Whisper to write punctuation (zh/ja conversation often
        # comes out without any). initial_prompt conditions the first window, hotwords every one.
        for k in ("initial_prompt", "hotwords"):
            if p.get(k):
                extra[k] = str(p[k])
        if req.clips:  # one speaker turn per clip: Whisper cannot merge words across turns
            extra["clip_timestamps"] = [round(t / 1000, 3) for c in req.clips for t in c]
        batch_size = int(p.get("batch_size", 0))
        if batch_size > 0 and not req.clips and op == "asr.transcribe":
            # M5: VAD chunks of <= 30 s decoded in batches (BatchedInferencePipeline)
            from faster_whisper import BatchedInferencePipeline

            if self.batched is None:
                self.batched = BatchedInferencePipeline(self.model)
            segments, info = self.batched.transcribe(
                audio.mean(axis=1),
                language=lang,
                word_timestamps=True,
                batch_size=batch_size,
                beam_size=int(p.get("beam_size", 5)),
            )
        else:
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

    def offload(self) -> None:
        """Weights to CPU memory (3.6 GB of GPU memory freed in 2.1 s; back in 0.5 s, against
        9.7 s to load from the volume)."""
        if self.model is not None and self.model.model.model_is_loaded:
            self.model.model.unload_model(to_cpu=True)

    def restore(self) -> None:
        if self.model is not None and not self.model.model.model_is_loaded:
            self.model.model.load_model()

    def run_batch(self, op: str, reqs: list[TranscribeReq], ctx: RunCtx) -> list[WordTimeline]:
        if op != "asr.backcheck" or len(reqs) < 2:
            return [self.run(op, r, ctx) for r in reqs]
        self._ready(ctx)
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


# ---------------------------------------------------------------- punctuation transfer

SENTENCE_MARKS = ".?!"


def transfer_punct(words: list[Word], donor: list[Word], window_ms: int = 60_000,
                   margin_ms: int = 5_000) -> list[Word]:  # fmt: skip
    """Copy the sentence-final punctuation of `donor` (an ASR that punctuates, e.g. Whisper)
    onto `words` (an ASR that does not, e.g. PhoWhisper), where the two agree on the text.

    Characters are matched with case and diacritics folded, window by window along the
    timeline (both sides have word times), so a long video stays linear. A mark moves only
    when the donor word's last character matches the last character of one of our words:
    where the two ASRs heard different words (numbers as digits vs as words, an invented
    phrase), nothing moves.
    """
    import difflib

    from dub.ops.align import _norm_chars

    out = [w.model_copy() for w in words]
    if not words or not donor:
        return out
    end = max(words[-1].end_ms, donor[-1].end_ms)
    for ws in range(0, end + 1, window_ms):
        we = ws + window_ms
        tgt: list[str] = []
        owner: list[int] = []
        for i, w in enumerate(words):
            if ws - margin_ms <= w.start_ms < we + margin_ms:
                for _, c in _norm_chars(w.w, strip_marks=True):
                    tgt.append(c)
                    owner.append(i)
        src: list[str] = []
        marks: dict[int, str] = {}
        for w in donor:
            if not ws <= w.start_ms < we:
                continue
            chars = [c for _, c in _norm_chars(w.w, strip_marks=True)]
            if not chars:
                continue
            src.extend(chars)
            last = w.w.strip()[-1:]
            if last in SENTENCE_MARKS:
                marks[len(src) - 1] = last
        if not marks or not tgt:
            continue
        sm = difflib.SequenceMatcher(None, src, tgt, autojunk=False)
        s2t = {a + k: b + k for a, b, n in sm.get_matching_blocks() for k in range(n)}
        for si, mark in marks.items():
            ti = s2t.get(si)
            if ti is None:
                continue
            wi = owner[ti]
            if ti + 1 < len(owner) and owner[ti + 1] == wi:
                continue  # not the end of our word: the two ASRs split the words differently
            if out[wi].w[-1:] not in SENTENCE_MARKS + "…":
                out[wi] = out[wi].model_copy(update={"w": out[wi].w + mark})
    return out


class PunctTransferASR:
    """Words and times from one model, sentence punctuation from another (M7 experiment).

    PhoWhisper hears Vietnamese best but writes no punctuation, and SaT then has to guess
    sentence ends from bare words (E4: Vietnamese F1 0.711 unpunctuated, 0.987 punctuated).
    Whisper large-v3 punctuates but mishears more, and sometimes invents sentences; its
    punctuation is only transferred where both heard the same words.
    """

    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self.words_asr = FasterWhisperASR(manifest)
        self.punct_asr = FasterWhisperASR(manifest)

    def _ctx(self, ctx: RunCtx, prefix: str) -> RunCtx:
        import dataclasses

        p = ctx.params
        params = {k: v for k, v in p.items() if not k.startswith("punct_")}
        if prefix:
            params.update({k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)})
        return dataclasses.replace(ctx, params=params)

    def offload(self) -> None:
        self.words_asr.offload()
        self.punct_asr.offload()

    def restore(self) -> None:
        pass  # each model comes back on its next use (FasterWhisperASR._ready)

    def run(self, op: str, req: TranscribeReq, ctx: RunCtx) -> WordTimeline:
        heard = self.words_asr.run(op, req, self._ctx(ctx, ""))
        if op != "asr.transcribe":
            return heard  # back-checks only need the words
        donor = self.punct_asr.run(op, req, self._ctx(ctx, "punct_"))
        return WordTimeline(lang=heard.lang, words=transfer_punct(heard.words, donor.words))
