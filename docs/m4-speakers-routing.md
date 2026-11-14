# M4: speakers, routing, second engine

Videos with several people: find who speaks when, give each person their own cloned voice,
keep the voices apart, and make the translation address people the way the target language
expects (Vietnamese "con/mẹ", "cháu/ông"... instead of a generic "tôi/bạn"). Every default
below comes from a measurement; the scripts are in [`experiments/`](../experiments/README.md).

## What changed

| Area | Before (M3) | M4 |
|---|---|---|
| Speakers | `single_speaker`: one cloned voice for the whole video | `ecapa_cluster`: SpeechBrain ECAPA embeddings of 1.5 s windows, average-linkage clustering (threshold 0.85, clusters under 5% absorbed), turns re-cut at real pauses |
| ASR with several speakers | one pass over the whole track | turn by turn (`clip_timestamps`): no word crosses a speaker change |
| Speaker of a word | nearest turn | words well inside a turn are anchors; the rest follow their nearest anchored neighbour |
| Reference voice | the 3–10 s line with the best ASR confidence | the most typical pure line (purity ≥ 0.95, 3–10 s): closest to the speaker's centroid |
| QA of a take | back-check CER in the risk score | gate per take: CER ≤ 0.15, the voice closer to its own speaker than to any other, similarity ≥ 0.5 of the speaker's own, ending heard; `final` tries up to 3 seeds |
| Forms of address | whatever the LLM wrote | rule checker per line + whole-script consistency, listener inferred from vocatives then turn-taking, flagged windows translated again with the problem stated (final, ≤ 2 rounds); rewrites must keep the terms |
| Cast sheet | written by the LLM | user corrections win and lock the row: `dub edit … cast`, the review page's cast panel, `dub run --confirm-cast` |
| TTS engines | OmniVoice | + VieNeu-TTS v3 Turbo (vi, en; `tts2` env; candidate) |
| Pod envs | each venv a full copy | symlinks into one uv cache on the volume: torch stored once |

## Diarization (E8)

**Question:** which speaker model and settings find the right people, on material with known
truth? Pyannote needs a Hugging Face token for its gated models, so the first candidate was
SpeechBrain's ECAPA speaker encoder (Apache-2.0, no token) with plain clustering.

Material:
- **drama4**: a 4-person family scene (`experiments/drama.json`, 42 lines), each character a
  different cloned voice, quick turn-takes down to 150 ms, music bed;
- **interview2**: the 2-voice interview clip with known onsets;
- **ami4**: AMI meeting ES2004a, first 10 minutes: a real 4-person meeting (CC BY 4.0).

Frame-level scores (10 ms, best one-to-one speaker mapping):

| Setting | drama4 found / confusion | interview2 found / confusion | ami4 found / confusion / missed |
|---|---|---|---|
| `single_speaker` | 1 of 4 / 69% | 1 of 2 / 40% | 1 of 4 / 53% / 0% |
| ECAPA, threshold 0.70 | 5–6 / 10–11% | 2 / 0% | 5–9 / 12–48% / 40% |
| ECAPA, threshold 0.80 | 5 / 10% | 2 / 0% | 4–5 / 8–12% / 39% |
| **ECAPA, threshold 0.85, min share 5%** | **4 / 2.8%** | **2 / 0%** | **4 / 9.4% / 39%** |
| ECAPA, threshold 0.90 | 4 / 2.8% | **1** / 40% | 4 / 8.8% / 39% |
| ECAPA, true speaker count given | 4 / 2.8% | 2 / 0% | 4 / 8.8% / 39% |

**Decision:** threshold 0.85, clusters under 5% of the speech absorbed by their nearest
neighbour: the only setting with the right count on all three. Below it the drama splits
into 5–6 people; at 0.90 the two interview voices merge. The 39% "missed" on AMI is the
energy VAD missing quiet far-field speech, not a clustering error; overlapped speech is
not handled (each frame has one speaker).

Two fixes came from full runs of the drama, not from the frame scores:
- **Turns re-cut at pauses.** With quick turn-takes the window grid moved speaker changes by
  up to 1 s. Turns are now rebuilt from the VAD pieces (merge gap 60 ms): each piece ≥ 400 ms
  goes to its nearest centroid.
- **Turn-by-turn ASR.** Whisper times a phrase's first word early, into the pause, and merges
  words across a turn change ("…promises, Dad"): the vocative landed in the previous
  speaker's line and the translator picked the wrong listener. Words are now assigned from
  anchors, and with more than one speaker the ASR runs per turn (`clip_timestamps`, VAD
  filter off). Speaker swaps on the drama: 3 → 0.

Each line also gets a **purity** (share of its time spoken by its speaker); `metrics.json`
reports `units_impure` and `purity_p10`, and the risk score uses it.

## Voices and the QA gate

Every line is embedded (`audio.embed`, ECAPA on the vocals). A speaker's centroid is the
mean of their lines; the reference clip is the pure 3–10 s line closest to it, so a cough
or an overlap is not cloned.

The QA gate judges each take (`src/dub/ops/qa.py`, limits in `dub.yaml: qa`):

| Check | Fails when |
|---|---|
| words | back-check CER > 0.15 |
| truncation | less than 60% of the line's last quarter is heard |
| voice, relative | the take is closer to another speaker's centroid than to its own (+0.05 margin) |
| voice, absolute | similarity < 0.5 × the speaker's own typical similarity |

Voice is not judged under 1.2 s (ECAPA is unreliable on very short clips). The first
version used an absolute floor of 0.75: cross-lingual clones sit at ~0.72 of the speaker's
own similarity (drama, p50), so it failed 29 of 54 lines and made `final` try three takes
for almost every line. The relative test catches what matters (a take that sounds like
someone else) without that.

Drama, en → vi, `final`: QA passed 97.8% of lines, voice similarity ratio p50 0.72,
p10 0.60, no truncation, back-check CER p50 0, p95 0.088.

## Forms of address (E10)

The cast sheet (`mt.context`) gives each pair of speakers their terms in the target language
("S1 > S2: self mẹ, other con"). After translation, `src/dub/ops/address.py` checks every
line:
- the listener: a vocative in the line ("Mom, …") wins, then turn-taking (the line before
  and the answer after); a question answered right away sets the listener;
- the terms: the expected self/other terms, or a generic pronoun (`address_generic` in the
  langpack: tôi, tao, tớ, mày, cậu, bạn) where the pair uses kinship terms. "mình"/"ta" are
  not generic in Vietnamese ("nhà mình" = our home);
- consistency over the whole script: the same pair keeps the same terms unless the cast
  sheet says the relationship changes (`since_ms`).

Flagged windows are translated again with the problem written into the request (`fix`), up
to two rounds in `final`. All three length variants are checked (the fit may pick any), and
a rewrite gets the terms and is dropped if it breaks them.

**Drama result (en → vi, final):** 42 lines, 4 characters (mother, son, daughter,
grandfather): **0 speaker swaps, 42 of 42 lines addressed correctly**, by the rule metric
(no generic pronoun, no wrong "X ơi" vocative) and by reading every line. Failure modes met
on the way, each with its fix: the vocative attached to the wrong speaker (turn-by-turn ASR),
the listener guessed wrong (listener rule in the prompt + `listener_issues`), "tôi" brought
back by the length rewrite (terms passed to the rewrite), the checker reading only the chosen
variant (all variants checked).

Corrections by the user: `dub edit VIDEO --to vi cast S2 gender=female name=Linda` or
`cast S1>S2 self_term=mẹ other_term=con`, or the cast panel of the review page. Only the
translation windows where those speakers appear are translated again (their cast rows are
part of the window's cache key). `dub run --confirm-cast` stops after the cast sheet and
prints it (`cast.md`: each speaker's talk time and first lines, to recognise them) so it can
be corrected before anything is translated.

## ASR per source language (E6)

**Question:** which ASR for each source language, and should it hear the mix or the separated
vocals? Material: the 11 real videos of M3 (first ~5 minutes, human subtitles). CER against
the subtitles (letters and digits only); subtitles are not always verbatim, so compare
columns rather than absolute values.

| Video | Model | Mix | Vocals |
|---|---|---|---|
| en-obama-2016 | Whisper large-v3 | 0.023 | 0.020 |
| en-obama-2009 | Whisper large-v3 | 0.018 | 0.017 |
| en-jfk-rice | Whisper large-v3 | 0.012 | 0.016 |
| es-maduro | Whisper large-v3 | 0.042 | 0.042 |
| es-wiki-cuenta | Whisper large-v3 | 0.001 | 0.002 |
| zh-vivo-interview | Whisper large-v3 | 0.096 | 0.096 |
| zh-polio | Whisper large-v3 | 0.251 | 0.251 |
| ja-digital-q | Whisper large-v3 | 0.117 | 0.110 |
| ja-mynaportal | Whisper large-v3 | 0.083 | 0.077 |
| vi-hcm-1945 (1945 archive) | PhoWhisper large | **0.120** (0.090 without numbers) | 0.122 (0.096) |
| | Whisper large-v3 | 0.155 (0.153) | **0.472** (0.476) |
| vi-pho-lai (TV news) | PhoWhisper large | 0.263 (0.123) | 0.205 (0.054) |
| | Whisper large-v3 | **0.044** (0.030) | 0.040 (0.025) |

What the Vietnamese numbers are made of (`e6_asr.py analyse`, largest word differences):
- **Numbers.** PhoWhisper writes them out ("một nghìn chín trăm bốn mươi" for "1940"),
  Whisper and the subtitles use digits. Without numbers on both sides PhoWhisper's CER on the
  news drops from 0.20 to 0.054 (vocals). Spelled-out numbers are what TTS reads anyway.
- **Extra passages.** On the news PhoWhisper hears ~100 words that neither the subtitles nor
  Whisper have ("hàng trăm tình nguyện viên…"): plausibly real speech the subtitles skip,
  not verified by listening.
- **Hallucination.** On the 1945 archive Whisper, given the separated vocals, replaces whole
  sentences of the speech with YouTube outros ("hãy subscribe cho kênh…"): CER 0.47.
  PhoWhisper does not.

Sentence segmentation with each model as the ASR (`dub bench seg --use
asr.transcribe=whisper_large_v3`, F1 at ±1 word against the M3 references):

| Video | PhoWhisper | Whisper large-v3 |
|---|---|---|
| vi-hcm-1945 | **0.766** (P 0.73, R 0.80) | 0.672 (P 0.54, R 0.89) |
| vi-pho-lai | 0.627 (P 0.62, R 0.64) | **0.732** (P 0.61, R 0.91) |

A draw on average (0.70 each). Whisper's punctuation splits more (recall up, precision
down); neither model fixes Vietnamese segmentation, which stays the weakest language of M3.

**Decisions:**
- The ASR keeps hearing the **mix**: vocals help on 6 of 13 rows (mostly by < 0.01), hurt on
  4, change nothing on 3, and triggered the worst failure (the hallucination); the mix also
  lets the ASR run while the separation runs.
- **Vietnamese keeps PhoWhisper first**, Whisper as the fallback: Whisper is better on clean
  news, but its failure mode (inventing sentences) is worse for dubbing than PhoWhisper's
  (spelled-out numbers, possibly extra true words).
- **Japanese stays on Whisper large-v3.** Kotoba-Whisper v2 (the distilled ja model) cannot
  serve the pipeline: faster-whisper's word alignment needs alignment heads the distilled
  decoder does not have, and crashes (`std::bad_alloc` in `find_alignment`). It stays a
  `candidate` manifest.

## The 50 GB volume, again

Installing the second TTS engine (its own env: VieNeu pins transformers 4.57, OmniVoice needs
5.x) hit the volume quota. Three envs held their own copy of torch 2.8 and the CUDA wheels
(~6 GB each), and the quota counts hardlinks as copies. Now the uv cache lives on the volume
and every venv is a tree of **symlinks** into it (`UV_LINK_MODE=symlink`), so each package is
stored once. Checked on the GPU before switching (cuBLAS matmul, cuDNN conv through
symlinked libraries). Result: **36.9 GB for 4 worker envs, cache, models and pipeline cache;
it was 41.5 GB for 3**. Rules: never `uv cache clean`/`prune` that cache; venvs are rebuilt
from scratch when their requirements change (cheap with symlinks); bytecode is compiled at
install time (first imports otherwise wrote thousands of `.pyc` files to the network volume).

## Not done in M4

- **Voice age/gender model** as cast-sheet evidence (the plan named an audeering model): not
  built. Gender and age still come from the dialogue (the LLM) and from the user.
- **Blind A/B listening tests**: not run. E1 compares engines with automatic measures only.
- **Local reference clips** (a reference per line instead of per speaker): not built; one
  typical clip per speaker passed QA on 97.8% of the drama's lines.
- **Pyannote / Sortformer** in E8: not compared (gated models; ECAPA met the criteria).
- **Overlapped speech**: each moment has one speaker; overlaps go to whoever dominates.
- **Kotoba-Whisper (ja)**: cannot run in the pipeline (see E6).

## Running it yourself

```bash
uv run dub run drama.mp4 --to vi --from en --remote --profile final --confirm-cast
uv run dub edit drama.mp4 --to vi cast S4 name=Grandpa age_band=senior
uv run dub run drama.mp4 --to vi --from en --remote --profile final
uv run python experiments/e10_address.py dub-workspace/jobs/<job>     # swaps + address
uv run dub run drama.mp4 --to vi --from en --remote --profile final \
    --use tts.synthesize=vieneu_v3_turbo                              # the second engine
```

`report.md` and `metrics.json` carry `speakers`, `units_impure`, `purity_p10`, `qa_passed`,
`voice_sim_ratio_p50/p10`, `ref_sim` per speaker and `address_flagged`; `units.json` has the
QA verdict, purity and address issue of every line.
