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
| Voice audition (4 candidate clips per speaker) | picked another clip (+0.07 on the test sentences), whole-job voice similarity unchanged (0.599 → 0.600) | no: off by default, available |
| Mix stage: levels per line, export in the background | 72 → 51 s on the 60-min final | yes |
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

Some projects pick the reference clip by how well it clones rather than by how typical it is
of the speaker. `voice.audition: N` tries that: the N most typical 3-10 s clips of each speaker
say the langpack's two calibration sentences (the same takes that measure the silence pad),
each take is back-checked and embedded, and among the intelligible clips (CER within 0.05 of
the best) the one whose takes are closest to the speaker's centroid becomes the reference.

A/B on the 60-minute speech, final profile, same pod:

| | First clip (most typical) | Audition of 4 |
|---|---|---|
| clip kept | 1 | 4 (test sentences: similarity 0.484 against 0.415) |
| voice similarity ratio p50 / p10, all 415 lines | 0.599 / 0.527 | 0.600 / 0.529 |
| QA passed | 94.7% | 94.0% |
| back-check CER p95 | 0.107 | 0.128 |

The gain on the two test sentences did not carry over to the job: on clean speech the
candidate clips clone about equally well, and two sentences per clip cannot tell them apart
(the line-to-line spread of similarity is larger than the differences between clips). The
audition is therefore off by default (`audition: 1`), kept for sources whose clips differ in
quality (noise, music, reverberation), where it can be measured again with
`--param dub.voice.audition=4`. The drama of M4 could not test it: in 2 minutes no speaker
has more than one 3-10 s line.

## Mix stage

On the first final with per-line levels the mix stage took 72 s, all in sequence: levels of
the 415 lines 19 s, tiles 5 s, concat 4 s, master 40 s, subtitles 8 s. Now each line's level
(and the original's) is measured as soon as the line is voiced (`audio.line_level`, cached
per line, off the critical path); the gains need every line of a speaker and are computed at
the end in no time; concat and master run in the background while the subtitles are
written. The stage takes **51 s**. The master (one loudness measurement plus an AAC encode
of the hour) is what is left; the AAC encoder is single-threaded.

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
language that is not evidence either way, so more evaluation videos were added before
deciding (next section).

### More evaluation videos: YODAS2

Commons has no more Vietnamese speech with subtitles (its Vietnamese subtitle pages are
translations of foreign videos, anthems and poems). YODAS2 (espnet/yodas2, CC BY 3.0) has
YouTube videos published under CC BY with the subtitles their uploaders wrote
(`experiments/yodas_videos.py`). Candidates were picked from the subsets' text files
(native-sounding titles and content, sentence punctuation, at least 5 minutes), and every
one is checked before it counts, because uploaded subtitles are often translations:

- Whisper must hear the expected language: two "Vietnamese" vlogs were in fact German and
  English with Vietnamese subtitles;
- the pipeline's own transcript must match the subtitles (CER ≤ 0.35). Two Chinese videos
  first failed at CER 0.50 and 0.36 because their subtitles are in traditional characters
  and Whisper writes simplified ones; references are now compared in simplified characters
  (OpenCC), which brought them to 0.26 and 0.10;
- the subtitles must be punctuated (≥ 15 reference sentences in 5 minutes); three videos
  whose subtitles are only partly punctuated got hand-placed sentence ends instead
  (`experiments/golden/`, rule in its README).

Baseline F1 (±1 word) of the added videos:

| Video | Kind | Check | F1 |
|---|---|---|---|
| vi 7cw7ARr0Woc | monk answering a question, spontaneous | vi, CER 0.11 | 0.553 |
| vi 1eyplFhZu5Y | model essays read aloud | vi, CER 0.02 | 0.860 |
| vi 1sDqjIs_4rU | written essay read aloud | vi, CER 0.20 | 0.622 |
| vi 6_HyZDSeLVY | travel vlog (hand labels) | vi, CER 0.21 | 0.769 |
| vi c6vHKZ9Uho0 | radio essay | vi, CER 0.08 | 0.673 |
| vi 7KJ6fpaw8vA | podcast | vi, CER 0.10 | 0.622 |
| zh fGzlkXYo0hI | political commentary | zh, CER 0.04 | 0.576 |
| zh v5DZN7rsLh4 | market analysis (hand labels) | zh, CER 0.15 | 0.806 |
| zh dgnjO1qSF3I | tech talk (traditional) | zh, CER 0.26 | 0.771 |
| zh t5IDTECWpWM | zodiac talk, spontaneous (hand labels) | zh, CER 0.07 | 0.667 |
| zh lLOvGz4qZWU | story narration (traditional) | zh, CER 0.10 | 0.587 |
| vi 3oIyhjGnVLA, eremzE9UQi4 | vlogs | heard de / en | not used |
| vi 2c8OawTG8TQ | book talk | 8 reference sentences (no punctuation) | not used |

Vietnamese now has 8 videos (2 Commons + 6), Chinese 7 (2 + 5).

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
