# M8: one pass, near-zero errors

Goal: a video goes in, the final dub comes out, with no human correction. The test case is
a 5-minute Chinese short drama from TikTok (burned-in subtitles, ten speaking parts, a song
under some scenes) dubbed into Vietnamese. Its first dub needed three or four rounds of
manual edits, and the listener still reported four kinds of error:

1. some passages still sound Chinese;
2. some Vietnamese is unclear;
3. some words are wrong;
4. some lines have the wrong voice (wrong person).

Each was traced to its causes with a measurement before anything was changed. Unless stated
otherwise: RTX PRO 4500 Blackwell (32 GB, $0.72/h), zh → vi, final profile.

## Trend table

| Change | Measured on the TikTok drama | Kept? |
|---|---|---|
| Who says each line read from the script (`text.speakers`) | speaker changes that change voice: 4% (default clustering), 22% (after the re-clustering loop) → 90%, precision 87%; B-cubed F1 0.42 / 0.54 → 0.81 | yes |
| Re-clustering finer when the cast sheet finds several people under one voice (M8 first try) | 2 → 5 voices but changes 22% → 24%; QA pass 0.91 → 0.84 | replaced by the above |
| Burned-in subtitles (OCR on the laptop) correct the transcript | 61 characters fixed; ASR vs subtitles CER 8.8% | yes |
| A line with no subtitle over it stays original | kept real answers in Chinese ("没谁啊") | only for songs and background voices now |
| Reference clip heard again by the ASR, cleanest of 3 candidates | every take of one voice began with 2 extra syllables from a clip with laughter | yes |
| Back-check text a short clip cannot hold is dropped | PhoWhisper turned a 600 ms "Gầy rồi." into 13 syllables | yes |
| OmniVoice's 100 ms fade → 10 ms | the fade ate the first and last phonemes of short lines | yes |
| Dub is the only audio track of the output video | players picked the original Chinese track | yes |
| Untranslated CJK characters sent back to the translator | guard against Chinese left in a Vietnamese line | yes |
| Short lines planned 1.6x longer (≤ 6 syllables, fading to 12) | 4-6 syllable lines understood 67% → 96% | yes |
| Accent of cross-lingual clones | see [accent](#chinese-accent) | (pending) |

## 1. Who says each line

### The measurement

Every subtitle cue of the drama (206 cues) was labelled with who says it, from the script:
question and answer, forms of address, who gives orders, the scene
(`experiments/golden/tiktok-7707275104924798249.speakers.json`, times and speaker only;
29 cues where the script does not tell are marked `?` and left out).
`experiments/phrase_speakers.py` scores any speaker assignment against them:

- **B-cubed** precision / recall / F1: precision falls when one voice carries several people,
  recall when one person gets several voices; no mapping of found to true speakers needed;
- **turn changes**: of consecutive labelled cues (gap < 2 s) where the speaker changes, the
  share that change voice (recall), and of the voice changes, the share that are real
  (precision). This is what a listener hears: "the same voice asks and answers".

| Assignment | Voices | B3 P | B3 R | B3 F1 | Change P | Change R |
|---|---:|---:|---:|---:|---:|---:|
| ECAPA clustering, threshold 0.85 (the default) | 2 | 0.27 | 0.97 | 0.42 | 0.67 | 0.04 |
| ECAPA clustering, 0.80 / 0.75 | 3 | 0.35 | 0.90 | 0.51 | 0.86 | 0.24 |
| ECAPA clustering, 0.70 | 4 | 0.36 | 0.91 | 0.52 | 0.86 | 0.24 |
| Finished job after the re-clustering loop (0.75) | 5 | 0.39 | 0.89 | 0.54 | 0.73 | 0.22 |
| Cues embedded and clustered, distance 0.7 | 35 | 0.57 | 0.34 | 0.43 | 0.44 | 0.96 |
| Cues embedded and clustered, distance 0.8 | 12 | 0.43 | 0.77 | 0.55 | 0.56 | 0.58 |
| Cues moved to the nearest centroid of the 0.75 clustering | 3 | 0.38 | 0.85 | 0.53 | 0.62 | 0.52 |
| **`text.speakers` (finished job)** | 6 | **0.74** | 0.89 | **0.81** | **0.87** | **0.90** |

### Why the voices could not do it

Mean ECAPA cosine between cues (long cues only), same person on the diagonal:

| | wet nurse | hiring woman | Steward Sun | husband |
|---|---:|---:|---:|---:|
| wet nurse | 0.32 | 0.23 | 0.29 | 0.15 |
| hiring woman | | 0.31 | 0.30 | 0.13 |
| Steward Sun | | | 0.33 | 0.18 |
| husband | | | | 0.28 |

The women of this drama are barely further apart from each other than from themselves:
no threshold separates them. The man is clearly apart. One-second cues are short for a
speaker embedding, but the lines of a quick exchange are that short. VoiceStudio's
recovery of a rapid two-speaker exchange (cluster the phrases in two, accept when the mean
cosine within beats the mean between by 0.12) would decline here (0.32 vs 0.23: 0.09).

### What reads it instead

`text.speakers` (Claude, `configs/prompts/text_speakers.v1.md`, `dub/ops/speakers.py`):

1. **Lines**: the subtitle cues when the subtitles carry the dialogue (one cue is almost
   always one speaker), plus the speech no cue covers; otherwise the transcript cut at
   pauses of 300 ms, sentence ends and diarizer turn changes. A cue's line spans the ASR
   words aligned to it **by characters** (`hardsubs.align_words`): cue times come from
   frames at 4 fps and from when the subtitle changed, a few hundred ms off the speech, and
   split by time a question took the first character of the answer.
2. **The LLM** names the characters (role, gender, kind: person / singing / background) and
   says who speaks each line. It sees the diarizer's voice label under each line and the
   gender of each voice (wav2vec2 age/gender model), told that a voice of the other gender
   is almost never the character, but that one label often holds two similar voices and
   misses quick exchanges. Windows of 300 lines go in order, each seeing the characters
   named before, so ids stay stable over a long video.
3. **The check**: if more than 15% of the lines would give a character a voice of the
   other gender (both sure), the answer is not used and the diarizer's turns stay.
   Otherwise the characters become S1, S2... and their lines the speaker turns.

Segmentation does not smooth these turns (a one-word answer between two lines of the
other person is real, not a diarization slip). The QA gate no longer fails a take for
"sounding like another speaker" when that speaker's centroid is within 0.8 of the line's
own (two similar voices). Lines whose speaker the LLM marked as a guess are flagged in
the review list.

On the drama: 6 people found in 217 lines (the wet nurse, the hiring woman, her husband,
Steward Sun, the master, a background song), 0 gender conflicts, 47 lines guessed. The
remaining errors are minor parts merged into a voice of the same gender and scene (two
servants and a gatekeeper under Steward Sun, the master's wife under the master) and a few
one-word answers.

## 2. Chinese in the dub

| Suspected cause | Measured | Result |
|---|---|---|
| Original voice left in the separated background | Chinese ASR on the bed alone (`experiments/leak_check.py`): 0 confident words, against 814 in the vocals stem | not the cause |
| Player playing the original track | the output had two audio tracks; some players pick the first | dub is now the only track (`--keep-original-audio` for both) |
| Reference text ≠ reference audio | OmniVoice continues its prompt: reference clips are now heard again by the ASR | fixed |
| Chinese characters left in a translation | guard: lines with CJK are sent back to the translator | fixed |
| Lines kept original because no subtitle covers them | 9 lines, among them real answers ("两", "没谁啊") | now only songs and background voices |
| Vietnamese spoken with the actor's Chinese accent | Whisper (language auto) heard 4 of 35 lines as Chinese, PhoWhisper understood them (CER 0.03-0.08) | see [accent](#chinese-accent) |

### Chinese accent

(pending: `experiments/accent.py`)

## 3. Unclear Vietnamese

### Short lines

Answers of one or two words ("Trai." for 男的, 240 ms) were the lines the listener could not
understand: the plan gave them their syllables at the average rate (4.7 syllables/s), and
OmniVoice, asked for exactly that, swallows them. Its own duration estimator does the
opposite: anything under 2 s is lengthened along a cube root.

`experiments/short_lines.py`: 12 lines of 1-6 syllables voiced at 1.0-2.5 times the linear
plan by the six voices of the drama and two E1 voices (English, Chinese), heard by
PhoWhisper (text the clip cannot hold dropped, see below). Share of takes with CER ≤ 0.15,
one voice with a leaking reference clip left out:

| Planned length | All | 1 syllable | 2-3 syllables | 4-6 syllables |
|---|---:|---:|---:|---:|
| 1.0 × linear | 0.32 | 0.00 | 0.19 | 0.67 |
| 1.3 × | 0.39 | 0.00 | 0.33 | 0.67 |
| **1.6 ×** | **0.47** | 0.00 | 0.31 | **0.96** |
| 2.0 × | 0.44 | 0.25 | 0.31 | 0.75 |
| 2.5 × | 0.43 | 0.17 | 0.33 | 0.71 |

Lines of 4-6 syllables want 1.6 times their linear length; more time makes them drawn out
and worse again. Lines of 2-3 syllables gain up to 1.3x and then stay at a third, and a
single syllable cannot be judged by an ASR at all: "Trai." comes back as "truyện", "trại",
"chuyện" at any length, a word out of context. Kept (`configs/langpacks/vi.yaml`):
`short_boost: 1.6` up to `short_units: 6`, fading to 1x at 12 syllables. Lines of three
syllables or fewer are no longer sent to the LLM to be shortened (one word less is worse
to hear than a line running 100 ms into the pause).

### Back-check on short clips

Whisper models invent text on very short clips. PhoWhisper heard a 600 ms take of
"Gầy rồi." as "cay rồi. hàng trăm triệu tập theo khuynh hướng vô sản trong...": the QA gate
failed good takes and spent retakes on them. A clip cannot hold more than about 8
syllables a second (10 characters for zh/ja, 5 words otherwise); when the transcript does,
only the run of consecutive words that fits and best matches the line is judged
(`qa.plausible`).

### Voice references

- Clips cut at quiet points (not inside a word), never next to another speaker's speech;
  a speaker without a 3-10 s line gets a pooled reference from their short lines.
- The reference transcript is what the ASR hears in the cut clip, not the line's text.
- Three candidate clips per speaker; the first heard within CER 0.15 of its line is used,
  else the closest. A clip with laughter gave every take of the master's lines two extra
  syllables ("sát nhến ...").
- OmniVoice's own 100 ms fade-in/out was cut to 10 ms: on a 600 ms line it ate the onset.

## 4. Wrong words

Burned-in subtitles are what the actors say, written by the producer. `dub.hardsubs_ocr`
reads them on the laptop (RapidOCR; band found on 12 sampled frames, then recognition only
when the band's picture changes); `hardsubs.correct` aligns them with the transcript
character by character, window by window, and the cue wins where they differ. Whisper's
transcript differed from the subtitles by 8.8% of the characters, mostly homophones that
change the meaning (城东前府 for 城东钱府, 乃至为 for 奶只喂); 61 characters were corrected.

## Ideas from VoiceStudio

VoiceStudio (AGPL-3.0; ideas only, no code) was read file by file. Taken: re-hearing the
reference clip, cutting references at pauses, NFC text for the TTS, the observation that
OmniVoice gives short texts much more time than a linear estimate (its own estimator boosts
anything under 2 s with a cube root), splitting a line where the speaker changes. Not
taken: their background repair (it puts original speech back under the dub), phrase
re-clustering (declines on this drama, see above). They have no OCR of burned-in subtitles
and no handling of songs.

## Residual risks

(pending)
