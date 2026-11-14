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
| Quality signals | speed-up, flags | + back-check CER and transcript per line (risk score, review page), `metrics.json` |
| Speaking rates | prior guesses | measured for OmniVoice (E7): vi 5.22 → 4.7, es 6.5 → 5.85 units/s |
| Exact-duration window | `safe_ratio` [0.92, 1.08] | [0.85, 1.10] (E3) |
| Draft TTS | 32 diffusion steps | 16 (E9); profiles can now override provider params |

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
| test clip, full pipeline on the pod, onset error p50 / max | 409 / 494 ms (all early) | **6 / 10 ms** (dubbed voice: 4 / 14 ms) |
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

## Speaking rate, squeeze range, diffusion steps (E7, E3, E9)

**Setup** (`experiments/tts_rate.py`): OmniVoice clones a voice from a reference in *another*
language, as when dubbing (an English female voice for vi/es/zh/ja/th, a Chinese male voice
for en and vi), and reads 10 NTREX sentences per language at six speaking rates, 0.7x to 1.25x
the langpack rate. 620 clips. Quality: CER of a back-check ASR (Whisper large-v3; PhoWhisper
for Vietnamese) and UTMOS (naturalness; trained on English, so compared within a language).

| rate (x langpack) | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.25 |
|---|---|---|---|---|---|---|
| vi CER (n=20) | 0.067 | 0.086 | 0.078 | 0.104 | 0.145 | 0.233 |
| vi UTMOS | 3.26 | 3.15 | 3.09 | 3.20 | 3.12 | 3.14 |
| es CER | 0.003 | 0.002 | 0.005 | 0.021 | 0.044 | 0.091 |
| es UTMOS | 4.03 | 3.91 | 3.83 | 3.66 | 3.65 | 3.48 |
| en CER / UTMOS | 0.02 / 3.76 | 0.04 / 3.52 | 0.02 / 3.64 | 0.08 / 3.57 | 0.03 / 3.47 | 0.06 / 3.51 |
| zh CER / UTMOS | 0.03 / 3.80 | 0.03 / 3.65 | 0.03 / 3.76 | 0.04 / 3.66 | 0.04 / 3.61 | 0.04 / 3.64 |
| ja CER / UTMOS | 0.01 / 4.00 | 0.01 / 3.96 | 0.02 / 3.92 | 0.03 / 3.87 | 0.03 / 3.82 | 0.03 / 3.74 |
| th CER / UTMOS | 0.04 / 4.00 | 0.05 / 4.02 | 0.04 / 3.94 | 0.04 / 3.94 | 0.04 / 3.84 | 0.05 / 3.77 |

**E7, natural rate.** Left to choose the length itself, cross-lingual OmniVoice speaks about
half as fast as people do (vi 2.3 syllables/s, CER 0.17): its own estimate comes from the
character rate of a reference in another script. So the pipeline always gives it a length.
Vietnamese loses words above ~4.7 syllables/s and Spanish loses naturalness above ~5.9, so
their langpacks now use **4.7** and **5.85**; English, Chinese, Japanese and Thai hold up to
1.0x and keep their rates.

**E3, safe squeeze range.** Slowing down to 0.7–0.8x of the rate costs nothing measurable;
speeding up past ~1.1x raises the CER (vi, es). `safe_ratio` is now **[0.85, 1.10]**: the engine
may make a line up to 1.18x longer or 1.10x shorter than its natural length before stretching
takes over. Stretching the natural take with rubberband to the same lengths kept the words
(CER 0.04–0.06 up to 1.1x) but sounded far worse (UTMOS 1.3–1.4 against 3.1–3.3): those
lengths needed 1.6–2.9x tempo changes from OmniVoice's slow natural take. Let the engine hit
the length; stretch only the last few percent.

**E9, 16 vs 32 steps** (same lines at 1.0x; speed on an idle GPU):

| | vi | en | es | zh | ja | th |
|---|---|---|---|---|---|---|
| CER 16 / 32 | 0.094 / 0.104 | 0.100 / 0.081 | 0.026 / 0.021 | 0.025 / 0.043 | 0.034 / 0.029 | 0.041 / 0.044 |
| UTMOS 16 / 32 | 3.15 / 3.20 | 3.74 / 3.57 | 3.73 / 3.66 | 3.63 / 3.66 | 3.88 / 3.87 | 3.82 / 3.94 |
| GPU s per s of audio, 16 / 32 | 0.136 / 0.255 | 0.120 / 0.225 | | 0.123 / 0.235 | | |

No consistent difference in quality; 16 steps is 1.9x faster. Drafts (and therefore the edit
loop) use 16 steps (`profiles.draft.params.omnivoice.num_step`), final keeps 32.

## The test clip end to end (en → vi, RTX 4090)

| | draft | final (LLM rewrites on) |
|---|---|---|
| units, longest | 13, 9.9 s | 13, 9.9 s |
| sentence F1 against the script (`dub bench seg`) | 1.000 | 1.000 |
| onset error against the true line onsets, p50 / max | 6 / 10 ms (dub 4 / 14 ms) | same |
| speed-up p50 / p95 / max | 1.08 / 1.15 / 1.16 | 1.01 / **1.08** / 1.09 |
| rewrites | 0 | 4 |
| back-check CER p50 / p95 | 0.02 / 0.16 | 0.02 / 0.21 |
| subtitle cues, too fast to read | 17, 0 | 17, 0 |
| end drift | 0 ms | 0 ms |

The highest-CER lines are real problems the review page now shows next to the line: "Lan làm
tôi nghe già quá" was heard as "lan làm tôi ngã giận quá". Fixing them is priority 3 (voice
quality, M4: take selection by back-check).

**Edit loop.** A one-line edit now also re-runs that line's back-check and the subtitles:
edit → render p50 3.3 s (M2: 2.3 s), edit → audible p50 4.6 s, still exactly one TTS call per
edit. Measures that kept it there: integer-ratio resampling in numpy instead of ffmpeg (an
ffmpeg call on a clip cost 350–390 ms on the pod), and back-checks running while the tiles
mix. `backcheck: false` in a profile trades the subtitle timing and CER for ~0.4 s per edit.

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

## Exit criteria

| Criterion (roadmap) | Result |
|---|---|
| no unit longer than 15 s | yes: by construction (DP), and 0 in every run |
| onset error p95 ≤ 150 ms | yes on the test clip (10 ms) and for en/es/vi/zh speech (≤ 150 ms); **no** for some Japanese and Thai chunks (p95 ~0.5 s) |
| speed-up p95 ≤ 1.15 | final profile 1.08; draft 1.15 (no rewrites in drafts) |
| segmentation F1 (design: ≥ 0.92 talks, ≥ 0.85 conversation, on hand-labelled real video) | read news: en/es ≥ 0.92 on speech; vi/zh/th below on speech (see E4); real videos still to be labelled (`dub bench seg`) |
| Thai line breaks: 0 errors | yes (0 of 5363 breaks inside a word or cluster) |

Not done in M3, and why:
- **E6** (ASR choice per source language, ASR on the mix vs the vocals) moves to M4, where ASR
  routing per language lives.
- **Per-voice rate model**: the rates above are per language for OmniVoice. A per-voice model
  needs natural target-language speech of the same voice to learn from, which a cloning TTS
  does not give; revisit with a second engine (M4).
- **LLM as a third E4 arm** (asking Claude to place sentence ends): not run. SaT is the default;
  the remaining errors are mostly ASR errors (dropped phrases, numbers read out).
- 60-minute measurements (throughput, re-export) stay with the M5 benchmark.

Cost of the M3 pod session: ~62 minutes of an RTX 4090 at $0.89/h ≈ $0.92.

## Running it yourself

```bash
uv run dub run video.mp4 --to vi --from en --remote
# report.md, "Sync and quality": onset error, speed-up, unit lengths, back-check CER, drift
uv run dub bench seg dub-workspace/jobs/<job> --golden video.sentences.txt
```

`--golden`: the sentences actually spoken in the video, one per line, in the source language.
The command prints precision, recall and F1 of the job's sentence boundaries (±0 and ±1 word).
