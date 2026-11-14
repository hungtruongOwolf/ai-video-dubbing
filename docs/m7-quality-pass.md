# M7: quality pass

After M6 the code of eleven open-source dubbing projects was read for ideas worth taking,
and every idea was measured before it was kept. Unless stated otherwise: RTX PRO 4500
Blackwell (32 GB, $0.72/h) in EU-RO-1, the 60.1-minute test speech of M5 (one speaker),
en → vi, final profile.

## Trend table

| Experiment | Result | Kept? |
|---|---|---|
| Separation: mix loaded at the model's 44.1 kHz (was 48 kHz) | vocals SI-SDR 20.93 → 21.51 dB, bed 12.42 → 13.02 dB | yes |
| Separation: autocast (fp16 where safe) | same SI-SDR to 0.01 dB; 146.8 → 88.5 s per hour | yes |
| Separation: `batch_size` (kept in M5) | ignored by audio-separator for RoFormer; the M5 "speed-up" was noise | removed |
| Silence compensation for OmniVoice (per-voice pad) | LLM rewrites 140 → 25; speech 4.82 → 4.31 syl/s (p90 6.11 → 4.90); CER p95 0.136 → 0.107 | yes |
| Per-line loudness from the original vocals | raw TTS levels uncorrelated with the original (r = −0.08), 4 dB line-to-line mismatch (p90) now corrected | yes |
| Voice audition (4 candidate clips per speaker) | see [below](#voice-audition) | see below |
| Mix stage: levels per line, export in the background | see [below](#mix-stage) | yes |
| Spoken numbers (pronunciation layer) | OmniVoice reads vi numbers right in 10 of 11 lines | not needed |
| Segmentation trials (6) | each helps one video and hurts another | none kept |
| Worker restarts, versioned result keys, single-flight fix | robustness, see [below](#robustness) | yes |

## Reference projects

Read at the code level (call order, models, how each step is done): VideoLingo, WhisperX,
pyvideotrans, GPT-SoVITS, SoniTranslate, open-dubbing, ViDubb, Linly-Dubbing, YouDub-webui,
OpenCreator, voice-pro. None is a better base than this pipeline for what it is built for
(any-to-any languages, exact timing, cached reruns and edits), so ideas were taken one by one:

- **taken and measured**: separation at the model's own sample rate, per-line loudness
  following the original, picking the reference clip by how well it clones (voice
  audition), punctuated prompts for Whisper (zh/ja), punctuation from a second ASR (vi);
- **not needed**: a pronunciation layer for numbers (the TTS reads them; the translation
  prompt already asks for spoken forms), timing borrowing changes (M3's replay showed the
  late-end p95 at 0);
- **left for later**: subtitle break penalties (VideoLingo), rotating reference clips across
  retakes (at most 11 of 415 lines fail on voice after retakes), rewrite candidates and a
  window-level length fix (only 25 rewrites remain, below).

## Separation

`experiments/sep_quality.py`: six synthetic mixes (the five F5-TTS reference recordings,
joined, 60 s, over three public-domain pieces from Wikimedia Commons: piano, orchestra,
electronic; music 6 and 12 dB below the speech), SI-SDR of the separated vocals against the
clean speech and of the instrumental against the music, plus one separation of the
60-minute speech for speed.

| Setting | Vocals SI-SDR | Bed SI-SDR | 60 min |
|---|---|---|---|
| 48 kHz, fp32 (M5) | 20.93 dB | 12.42 dB | 152.7 s |
| 44.1 kHz, fp32 | 21.51 dB | 13.02 dB | 146.8 s |
| **44.1 kHz, autocast** | **21.51 dB** | **13.02 dB** | **88.5 s** |
| 44.1 kHz, native fp16 | 21.51 dB | 13.02 dB | 86.1 s |

- The model was trained at 44.1 kHz; feeding it 48 kHz shifts every frequency it sees. All
  six mixes improve, by 0.40-0.68 dB (vocals) and 0.42-0.73 dB (bed).
- Autocast is really on: a 60 s excerpt differs from fp32 by at most 1.8e-4 (fp32 run twice:
  bit-identical). After the `batch_size` lesson below, a setting that changes nothing in the
  output is checked before it is believed.
- Native fp16 is within noise of autocast and keeps more operations in half precision;
  autocast is the default.
- **Correction to M5:** `batch_size: 8` was kept in M5 for a 155 → 117 s speed-up with
  bit-identical vocals. audio-separator 0.48 ignores `batch_size` for RoFormer models; the
  output was identical because nothing changed, and the times were run-to-run variation.

## Timing: the silence inside OmniVoice's takes

OmniVoice generates exactly the duration it is asked for, but part of it is silence (its
post-processing trims long silences and pads 0.1 s each side). After our trim, the speech
was 90% of the requested length at p50 (76% for lines under 2 s), with ~340 ms of lead
silence: every line was spoken faster than planned (4.82 syllables/s at p50, 6.11 at p90,
for a planned 4.7).

That also explained most LLM rewrites. A rewrite starts when a line's speed-up exceeds 1.10;
for an exact engine the speed-up after the take is `planned / voice`, so a take that is 90%
voice reads as 1.11 even when the text fits its slot. In the M5 final, 101 lines were
rewritten (140 calls), yet only 14 lines of the whole job had no variant within 1.10× of
their slot at the planned rate; the 62 lines rewritten once had short variants at a median
0.81 of their slot.

**Fix:** before voicing, each voice says the langpack's two `calibration` sentences at their
predicted length; the median silence trimmed off is added to every later request for that
voice (385 ms for the test speech). Results, same video, final profile:

| | M5 final (RTX 4090) | Now (RTX PRO 4500) |
|---|---|---|
| LLM rewrite calls | 140 | **25** |
| speech rate p50 / p90 (syllables/s; plan 4.7) | 4.82 / 6.11 | **4.31 / 4.90** |
| speed-up p95 / max | 1.16 / 1.52 | **1.07 / 1.20** |
| back-check CER p95 | 0.136 | **0.107** |
| QA passed | 92.3% | **94.7%** |
| voice similarity ratio p50 | 0.606 | 0.599 |
| wall time | 844 s | 795 s |

`experiments/tts_silence.py` measured the same effect on the five reference voices (voice
0.94-0.985 of the request as is; trimming the reference clip did not help, the pad did).

## Voice: level

OmniVoice scales its output to the reference clip's RMS: every line of a speaker comes out at
about the same level. On the test speech the raw TTS levels do not follow the original at
all (correlation −0.08 between the line levels), with a line-to-line mismatch of 4 dB (p90)
that one overall gain cannot fix.

`ops/level.py` measures the active speech level (20 ms frames, pauses gated out) of each
dubbed line and of the original line in the separated vocals, and gives the line the gain
that brings it to the original's level: relative to the speaker's median (at most 10 dB from
it, so bleed in one slot cannot make a line jump), the gain capped at 24 dB. The mix applies
`Placement.gain_db`; the master's single gain and limiter come after, so the dynamics stay.

## Voice audition

(results pending)

## Mix stage

(results pending)

## Segmentation trials

F1 at ±1 word on the 11 real videos of M3 (`experiments/real_videos.py`). Baseline with the
new separation: mean 0.806 (unchanged).

| Trial | Changed videos | Kept? |
|---|---|---|
| vi: punctuation from Whisper large-v3 transferred onto PhoWhisper's words (`phowhisper_punct`) | vi-hcm-1945 0.766 → 0.727, vi-pho-lai 0.629 → 0.706 | no |
| SaT min sentence length 2 words | none | no |
| SaT min sentence length 3 words | vi-hcm-1945 0.766 → 0.753, zh-polio 0.830 → 0.846 | no |
| zh: punctuated `hotwords` for Whisper | zh-polio 0.830 → **0.964**, zh-vivo 0.762 → 0.684 | no |
| zh: punctuated `initial_prompt` | zh-vivo 0.762 → 0.716 | no |
| ja: punctuated `hotwords` | ja-digital-q 0.828 → 0.832, ja-mynaportal 0.852 → 0.828 | no |

Every trial helps one video of a language and hurts the other. With two videos per
language that is not evidence either way; the defaults stay, and the variants stay
available (`--use asr.transcribe=phowhisper_punct`, `--param whisper_large_v3.hotwords=...`).
Deciding them needs more evaluation videos with human subtitles in vi and zh.

## Spoken numbers

11 of 418 lines of the test speech still had digits after translation (the prompt asks for
spoken forms). The back-check heard OmniVoice read them: "15" → mười lăm, "140" → một trăm
bốn mươi, "11 tháng 9" → mười một tháng chín, "3D" → ba đê, "2014" → hai không mười bốn (the
colloquial year). One was wrong: "102" → hăm hai. A pronunciation layer is not worth its
cost now; the syllable count already estimates one syllable per digit.

## Robustness

- **Worker restarts.** dubd watches every ready worker; a process that exits (CUDA fault,
  killed for memory) is started again, at most 3 times in 15 minutes. A refused or cut
  connection is `kind: unavailable`, and the core waits for the worker up to ~5 minutes
  instead of retrying twice within 10 seconds.
- **Versioned result keys.** A manifest names the packages its output depends on
  (`libs: [audio-separator, torch]`); their pinned versions in `pod/envs/<env>.txt` are part
  of the provider version, so an upgrade redoes their cached results without a manual
  `impl_version` bump. A test checks that each named package is pinned.
- **Single flight.** A waiter whose job was stopped cancelled the shared future through
  `asyncio.wrap_future`: the owner's release raised `InvalidStateError` and every other
  waiter was cancelled too. Waiters now shield it (regression test included).
- **A/B of pipeline settings.** `--param dub.SECTION.KEY=VALUE` overrides a dub.yaml setting
  for one run; `bench speed --json` carries the job's quality metrics next to the timings.

## Cost

(pending)
