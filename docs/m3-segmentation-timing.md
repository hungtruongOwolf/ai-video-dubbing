# M3: segmentation and timing

Priorities 1 (sentence segmentation) and 2 (timing) of the project. Every default changed in
M3 comes from a measurement below; the scripts are in [`experiments/`](../experiments/README.md).

## What changed

| Area | Before (M2) | M3 |
|---|---|---|
| Sentence boundaries | punctuation rules | SaT `sat-12l-sm` probabilities per word (`text.boundaries` op); rules as fallback |
| Long sentences | DP on commas and pauses | same DP, cut cost = −log(p) − pause bonus |
| Unit edges | ASR word times | snapped onto the speech of the separated vocals (energy VAD) |
| TTS clip | placed as returned | trimmed to the voice before fitting |
| Audio extraction | `.mka` for every source | `.m4a` for MP4/MOV sources (keeps the AAC priming edit list) |
| Subtitle layout | split on spaces, or anywhere for zh/ja/th | per script: BudouX phrases + kinsoku (zh/ja), dictionary words (th), RLM (RTL), Netflix widths |
| Subtitle timing | spread by character count | follows the voice: back-check ASR words of each dubbed line |
| Quality signals | speed-up, flags | + back-check CER per line (in the risk score), `metrics.json` |

## Segmentation (E4)

**Question:** can a multilingual sentence model (SaT, wtpsplit, MIT) replace punctuation rules,
including on ASR text without punctuation, and which size and threshold?

### Text arm

All 1997 NTREX-128 sentences (123 news documents) in six languages, turned into ASR-like token
streams: as written (Whisper writes punctuation for most languages) and without punctuation or
capitals (PhoWhisper, Thai ASR). Boundary F1 with ±1 word tolerance; end of text excluded.
Run on the pod's GPU.

| F1 (with / without punctuation) | en | vi | th | zh | ja | es |
|---|---|---|---|---|---|---|
| punctuation rules | 0.934 / 0 | 0.942 / 0 | 0.052 / 0 | 0.883 / 0 | 0.830 / 0 | 0.925 / 0 |
| SaT 3 layers @0.25 | 0.982 / 0.589 | 0.976 / 0.519 | 0.676 / 0.631 | 0.936 / 0.627 | 0.910 / 0.798 | 0.981 / 0.634 |
| **SaT 12 layers @0.25** | **0.986 / 0.809** | **0.987 / 0.711** | **0.852 / 0.823** | **0.952 / 0.739** | **0.926 / 0.864** | **0.984 / 0.825** |

- Macro F1 over the 12 cells, SaT-12l: 0.871 at 0.2, 0.872 at 0.25–0.4, 0.870 at 0.5. Flat, so
  the wtpsplit default 0.25 stays. SaT-3l peaks at 0.772.
- The rules find nothing without punctuation, and almost nothing in Thai, which ends sentences
  with a space.

### Speech arm

The same text read aloud: OmniVoice reads 12 documents x 4 sentences per language (two sentences
per call, 350 ms between calls); the langpack's ASR transcribes them (Whisper large-v3;
PhoWhisper for Vietnamese); the pipeline's own segmenter cuts the transcript.

| F1 | en | vi | th | zh | ja | es |
|---|---|---|---|---|---|---|
| sentences, rules | 0.845 | 0.085 | 0.458 | 0.340 | 0.520 | 0.880 |
| sentences, SaT @0.25 | 0.921 | 0.626 | 0.711 | 0.677 | 0.870 | 0.933 |
| sentences, SaT @0.1 | 0.886 | 0.621 | 0.725 | **0.778** | 0.901 | 0.935 |
| dub units, pauses only (duration DP, no text model) | 0.444 | 0.246 | 0.492 | 0.073 | 0.345 | 0.542 |
| dub units, SaT @0.25 | 0.946 | 0.650 | 0.744 | 0.676 | 0.861 | 0.933 |

- Chinese is better with a lower threshold in both arms (speech +0.10, text +0.015), so the
  SaT manifest uses `lang_threshold: {zh: 0.1}`. Other languages keep 0.25.
- Vietnamese stays low, and not because of SaT: on clean unpunctuated text it reaches 0.71. On
  speech, PhoWhisper writes no punctuation, reads numbers out ("832" becomes "tám trăm ba mươi
  hai"), and sometimes drops a whole phrase together with the sentence end it contained.
- Also tried: adding the pause after each word to the SaT logit (β × pause, β = 1–3). No gain in
  any language (±0.01): SaT is already confident where the pauses are.

### Decision

`text.boundaries` routes to `sat_12l_sm` (threshold 0.25, Chinese 0.1), with `punct_rules` as the
fallback. It runs in the `tts` worker env next to OmniVoice: the network volume's quota counts
every venv's copy of torch (~5 GB), so a separate env did not fit in 50 GB.

**Against the design targets** (F1 ≥ 0.92 for talks, ≥ 0.85 for conversation, measured on hand-
labelled real videos): on read news, English, Spanish, and Vietnamese/Chinese text with
punctuation reach them; Japanese is close; Thai, unpunctuated Vietnamese and Chinese speech do
not. The real-video check needs your own labelled videos: `dub bench seg` (below).

## Slot onsets (timing)

A test clip with known line onsets (two Windows SAPI voices with a music bed, 10 lines; the
generator records where each line starts) and the 72 chunks of the E4 speech arm.

| | Whisper word times | snapped (M3 rule) |
|---|---|---|
| test clip, onset error p50 / max | 409 / 494 ms (all early) | 36 / 109 ms |
| E4 speech, onset error p50 / p95, en | 340 / 500 ms | 30 / 150 ms |
| E4 speech, es | 345 / 500 ms | 0 / 30 ms |
| E4 speech, ja | 290 / 540 ms | 0 / 510 ms |
| E4 speech, vi (PhoWhisper) | 90 / 120 ms | 0 / 10 ms |

Whisper places the first word of a phrase 300–500 ms early, inside the pause before it. The
first snapping rule (move to an onset within −400/+250 ms) could not reach the true onset; the
M3 rule moves an edge that lies in silence to the nearest speech (up to 800 ms), and an edge
inside speech to the edge of its region (up to 400 ms). A few Japanese and Thai chunks still
start ~0.5 s off; their Whisper word times are less regular.

The same clip exposed a 21 ms shift of the whole timeline: copying an MP4's AAC track into
Matroska drops the edit list that hides the encoder's 1024 priming samples. MP4/MOV sources
are now extracted to `.m4a`, which decodes exactly like the original.

## Subtitle layout (Thai, Chinese, Japanese)

Every NTREX sentence laid out as one cue item (`experiments/subs_layout.py`); every break
between lines and between cues is checked.

| | breaks | inside a word / phrase | broken tone-mark cluster | kinsoku errors | over-wide lines |
|---|---|---|---|---|---|
| Thai, M0 (any character) | 5517 | 4326 | 1085 | – | 0 |
| Thai, M3 | 5363 | **0** | **0** | – | 0 |
| Chinese, M0 | 4150 | 2329 | – | 374 | 0 |
| Chinese, M3 | 4548 | **0** | – | **0** | 0 |
| Japanese, M0 | 5857 | 4694 | – | 686 | 0 |
| Japanese, M3 | 7051 | 116 | – | **0** | **0** |

- "Inside a word" for Thai uses PyThaiNLP's dictionary words; for Chinese and Japanese, BudouX
  phrases. M3 breaks only where both PyThaiNLP and BudouX allow it (Thai), so its zero is by
  construction; the M0 column shows what character-level breaking (libass-style) does.
- The 116 Japanese breaks inside a phrase are phrases longer than a whole line (16 characters),
  split by character as a last resort, still following kinsoku.

## Running it yourself

```bash
uv run dub run video.mp4 --to vi --from en --remote
# report.md, "Sync and quality": onset error, speed-up, unit lengths, back-check CER, drift
uv run dub bench seg dub-workspace/jobs/<job> --golden video.sentences.txt
```

`--golden`: the sentences actually spoken in the video, one per line, in the source language.
The command prints precision, recall and F1 of the job's sentence boundaries (±0 and ±1 word).
