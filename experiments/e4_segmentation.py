"""Experiment E4: sentence segmentation. Punctuation rules vs SaT (3 and 12 layers).

Two arms.

text (in the pod's `text` env, needs wtpsplit + GPU):
    All 1997 NTREX sentences in en, vi, th, zh, ja, es, document by document, turned into the
    token stream an ASR would give, with punctuation (Whisper-like) and without (PhoWhisper-
    like). Boundary F1 (+-1 word) for the rules and for SaT at a sweep of thresholds.

speech (in the pod's `core` env, through the running dubd workers):
    Real speech: OmniVoice reads NTREX documents two sentences per call; the langpack's ASR
    (Whisper large-v3, PhoWhisper for vi) transcribes them; the pipeline's own segmenter cuts
    them with the rules, with SaT, and with pauses only (a duration DP with no text model).
    Also measures the onset error of unit starts, ASR times vs VAD-snapped, against the
    known start of each synthesized chunk.

Usage (on the pod):
    envs/text/bin/python experiments/e4_segmentation.py text
    envs/core/bin/python experiments/e4_segmentation.py speech --threshold 0.25
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dub.bench.seg import boundary_f1, micro  # noqa: E402
from dub.contracts.v1 import BoundaryReq  # noqa: E402
from dub.ops.segment import rule_probs  # noqa: E402
from experiments.common import (  # noqa: E402
    LANGS,
    asr_like,
    data_dir,
    documents,
    joiner_for,
    run_worker,
    save_json,
    sentences_from,
)

SAT_MODELS = {
    "sat-3l-sm": "137da054051ad9f1eac42025f758db4ac9f22535",
    "sat-12l-sm": "d70c72a9331b2d5a9e82baad00c64964a23a09bb",
}
THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _streams(lang: str, punct: bool):
    out = []
    for doc in documents(lang):
        tokens, space = asr_like(doc, lang, punct)
        out.append((doc, tokens, space))
    return out


def _score(streams, lang, probs_per_doc, thr) -> dict:
    scores = []
    for (doc, tokens, space), probs in zip(streams, probs_per_doc, strict=True):
        pred = sentences_from(tokens, space, probs, thr, joiner_for(lang))
        scores.append(boundary_f1(pred, doc, lang, tol=1))
    return micro(scores).as_dict()


def run_text(models: list[str]) -> dict:
    import os

    from huggingface_hub import snapshot_download
    from wtpsplit import SaT

    from dub.providers.boundaries_sat import build_text, word_probs

    results: dict = {"rules": {}, **{m: {} for m in models}}
    data = {
        (lang, cond): _streams(lang, cond == "punct")
        for lang in LANGS
        for cond in ("punct", "nopunct")
    }
    for (lang, cond), streams in data.items():
        probs = [rule_probs(t) for _, t, _ in streams]
        results["rules"].setdefault(lang, {})[cond] = {"0.5": _score(streams, lang, probs, 0.5)}
        print("rules", lang, cond, results["rules"][lang][cond]["0.5"], flush=True)
    for name in models:
        path = snapshot_download(
            f"segment-any-text/{name}",
            revision=SAT_MODELS[name],
            cache_dir=os.environ.get("DUB_HF_CACHE"),
            allow_patterns=["config.json", "model.safetensors"],
        )
        sat = SaT(path)
        sat.half().to("cuda")
        for (lang, cond), streams in data.items():
            t0 = time.time()
            reqs = [
                BoundaryReq(lang=lang, words=t, space_before=s, joiner=joiner_for(lang))
                for _, t, s in streams
            ]
            built = [build_text(r) for r in reqs]
            char_probs = list(sat.predict_proba([text for text, _ in built]))
            probs = [
                word_probs(cp, text, ends)
                for cp, (text, ends) in zip(char_probs, built, strict=True)
            ]
            res = {str(t): _score(streams, lang, probs, t) for t in THRESHOLDS}
            results[name].setdefault(lang, {})[cond] = res
            best = max(res.items(), key=lambda kv: kv[1]["f1"])
            print(name, lang, cond, f"{time.time() - t0:.1f}s", "best", best, flush=True)
    return results


# ---------------------------------------------------------------- speech arm

ASR_FOR = {"vi": "phowhisper_large_ct2"}  # others: whisper_large_v3
DOCS_PER_LANG = 12
SENTS_PER_DOC = 4
GAP_MS = 350  # silence between two synthesized chunks


def _seg(lang, words, params, probs, regions):
    from dub.contracts.v1 import SegmentReq
    from dub.ops.segment import segment

    req = SegmentReq(lang=lang, words=words, turns=[], params=params, probs=probs, speech=regions)
    return segment(req)


def run_speech(threshold: float, langs: list[str]) -> dict:
    import os

    import numpy as np

    from dub.contracts.v1 import AudioReq, VadReq, Word
    from dub.core.config import load_config
    from dub.ops import media, vad
    from dub.ops.audio_io import read_wav, write_wav
    from dub.ops.media import voice_edges
    from dub.store import Workspace

    ws = Workspace.open(
        Path(os.environ.get("DUB_ROOT", "/workspace/dub")) / "ws", db_path=Path("/tmp/exp.sqlite")
    )
    cas = ws.cas
    cfg = load_config()
    results: dict = {}
    for lang in langs:
        t0 = time.time()
        docs = [d[:SENTS_PER_DOC] for d in documents(lang) if len(d) >= SENTS_PER_DOC][
            :DOCS_PER_LANG
        ]
        lp = cfg.langpack(lang)
        per_doc = {"rules": [], "sat": [], "rules_units": [], "sat_units": [], "pause_units": []}
        onset = {"asr": [], "snapped": []}
        for di, doc in enumerate(docs):
            # 1. speech: two sentences per TTS call, chunks joined with a short silence
            parts, truth, pos, sr = [], [], 0, 24000
            for ci in range(0, len(doc), 2):
                sep = "" if lang in ("zh", "ja") else " "  # Thai separates sentences with a space
                text = sep.join(doc[ci : ci + 2])
                res = run_worker(
                    "tts",
                    "omnivoice",
                    "tts.synthesize",
                    {
                        "unit_id": f"e4-{lang}-{di}-{ci}",
                        "text": text,
                        "lang": lang,
                        "speaker": "S1",
                    },
                )
                audio, sr = read_wav(cas.path(res["audio_key"]))
                mono = audio[:, 0]
                edges = voice_edges(mono, sr)
                truth.append(round((pos + (edges[0] if edges else 0)) * 1000 / sr))
                parts += [mono, np.zeros(sr * GAP_MS // 1000, dtype=np.float32)]
                pos += len(mono) + sr * GAP_MS // 1000
            tmp = cas.tmp_path(".wav")
            write_wav(tmp, np.concatenate(parts), sr)
            key = cas.put_file(tmp, move=True)
            a16 = media.to_mono16k(AudioReq(audio_key=key), cas)
            # 2. ASR (the langpack's model), boundaries, VAD
            tl = run_worker(
                "asr",
                ASR_FOR.get(lang, "whisper_large_v3"),
                "asr.transcribe",
                {"audio_key": a16.audio_key, "lang": lang},
            )
            words = [Word.model_validate(w) for w in tl["words"]]
            words = sorted((w for w in words if w.w.strip()), key=lambda w: (w.start_ms, w.end_ms))
            breq = BoundaryReq(
                lang=lang,
                words=[w.w for w in words],
                space_before=[w.space_before for w in words] if not lp.joiner else [],
                joiner=lp.joiner,
            )
            sat_p = run_worker(
                "text",
                "sat_12l_sm",
                "text.boundaries",
                breq.model_dump(mode="json"),
                params={"threshold": threshold},
            )["probs"]
            speech = vad.speech_regions(VadReq(audio_key=a16.audio_key), cas).regions
            params = lp.segment_params()

            rules = _seg(lang, words, params, rule_probs([w.w for w in words]), speech)
            sat = _seg(lang, words, params, sat_p, speech)
            sat_raw = _seg(lang, words, params, sat_p, [])
            pause = _seg(
                lang, words, params, [0.0] * len(words), speech
            )  # no text model: DP on durations and pauses
            per_doc["rules"].append(boundary_f1([s.text for s in rules.sentences], doc, lang))
            per_doc["sat"].append(boundary_f1([s.text for s in sat.sentences], doc, lang))
            for name, sg in (("rules_units", rules), ("sat_units", sat), ("pause_units", pause)):
                per_doc[name].append(boundary_f1([u.text for u in sg.units], doc, lang))
            # 3. onset error of the unit that starts each chunk
            for t in truth:
                near = [u for u in sat_raw.units if abs(u.start_ms - t) < 1000]
                if not near:
                    continue
                u = min(near, key=lambda u: abs(u.start_ms - t))
                snapped = next(x for x in sat.units if x.id == u.id)
                onset["asr"].append(abs(u.start_ms - t))
                onset["snapped"].append(abs(snapped.start_ms + 30 - t))  # 30 ms = snap pad
        out = {k: micro(v).as_dict() for k, v in per_doc.items()}
        for k, v in onset.items():
            if v:
                v = sorted(v)
                out[f"onset_{k}_ms"] = {
                    "p50": statistics.median(v),
                    "p95": v[min(len(v) - 1, round(0.95 * (len(v) - 1)))],
                    "n": len(v),
                }
        out["seconds"] = round(time.time() - t0)
        results[lang] = out
        print(lang, out, flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["text", "speech"])
    ap.add_argument("--models", default="sat-3l-sm,sat-12l-sm")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--langs", default=",".join(LANGS))
    a = ap.parse_args()
    out = data_dir() / "results"
    if a.arm == "text":
        save_json(out / "e4_text.json", run_text(a.models.split(",")))
    else:
        save_json(out / "e4_speech.json", run_speech(a.threshold, a.langs.split(",")))


if __name__ == "__main__":
    main()
