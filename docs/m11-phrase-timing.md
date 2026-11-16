# M11: Phrase timing

A dubbed line used to be one take fitted to its slot as a whole. The TTS paused at its own
places, a comma or between two sentences, and spoke on through the actor's pauses. After M10,
the drama still had 112 windows of 250 ms where the dub was silent while the actor spoke,
inside lines whose level matched the original's (median -1.2 dB). Each one is a mouth moving
with no voice.

## How it works (`dub.ops.phrases`)

The fix works on the signal only. No new translation and no new take: the cache and the takes
the QA gate passed are kept.

1. **The actor's phrases.** Speech regions of the separated vocals inside the slot (the job's
   VAD, 10 ms), joined across pauses shorter than 250 ms.
2. **The take's cut points.** Silences of the take of at least 80 ms. A cut there never splits
   a syllable: a stop consonant's closure is shorter.
3. **Matching.** Each pause of the actor is matched to the take's silence at the same share of
   speech. The matching is monotonic (dynamic programming) and allows a shift of at most 0.15.
   A silence of 150 ms or more is preferred: it is a real break. A pause with no silence near
   it stays unmatched.
4. **Re-timing.** The take is cut at the matched silences. Each group of pieces starts where
   its phrase of the actor starts and is sped up or slowed down, within 0.87-1.15, to end
   where the phrase ends.
5. **Long pauses the actor speaks through.** A silence of the take that matched no pause and
   is longer than 220 ms is shortened to 100 ms.

The line keeps its onset, which the timing fit placed. It never ends later than before or later
than its slot. Each piece is stretched with the same engine as the whole take (rubberband),
with 5 ms fades at the cuts.

**Heard again.** A re-timed take goes through the back-check and the voice check again. If a
take that was heard right is no longer heard right, the line keeps its take as it was. That
happened to 2 of the drama's 37 re-timed lines and 1 of the cartoon's 4. The subtitles and
word times use what the second back-check heard.

`units.json` flags the re-timed lines (`phrases`). `metrics.json` counts them:
`phrase_lines`, `phrase_pauses_matched`, `phrase_pauses_shortened`, `phrase_reverted`. In
`dub.yaml`: `timing.phrase_align`, `phrase_pause_ms`, `phrase_rate`.

## Measured

Each count is a 250 ms window inside a dubbed line's slot, comparing the dub with the original
(both are mixed with the same music bed, so only the voice differs). Same translations and
takes in every column.

| Drama (89 lines) | M10 follow-up | Phrase timing |
|---|---:|---:|
| Dub silent while the actor speaks, during the line | 112 | **63** |
| Dub silent after the line ended, the actor still speaking | 67 | 56 |
| Dub speaking over a silent actor | 151 | 134 |
| Mean sync error (take vs slot) | 292 ms | 263 ms |
| Lines re-timed / pauses matched / pauses shortened | | 35 / 20 / 59 |
| QA passed | 0.966 | 0.966 |

| Cartoon (10 dubbed lines) | M10 follow-up | Phrase timing |
|---|---:|---:|
| Windows under the original by 10 dB or more (whole video) | 15 (3.75 s) | **10 (2.5 s)** |
| Mean sync error | 271 ms | 190 ms |

The drama's remaining windows are spread over long lines (7-11 s) with many phrases. The most
are in a grieving line ("A Đậu mất rồi...") where the actor sobs louder than the dub speaks.
That is the emotion of the voice, not its timing (see M10, references chosen by emotion).

## Not done

- **Phrases cut at word boundaries.** Cut points are silences only. A pause of the actor where
  the take has no silence (no comma) stays unmatched. The back-check's word times could add
  cut points between words.
- **Phrase-aware translation.** The LLM could mark where its wording breaks, phrase for phrase
  with the source, so the dub's pauses fall where the actor's do by meaning, not by share of
  speech.
- **A wider shift for long lines** (0.15 now). Untested.
