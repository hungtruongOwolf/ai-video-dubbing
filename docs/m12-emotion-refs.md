# M12: Voices by delivery

A voice-cloning TTS copies more than the timbre of its reference clip: it copies how the clip
is said, its pitch, its liveliness, its loudness. Each speaker had one reference, the clip most
typical of their voice, so every line of a speaker was said in the mood of that clip. In the
drama, the wet nurse's clip was a grieving line. The M10 review measured the dub's
line-to-line pitch spread at 1/3 to 1/6 of the source's: every line said the same way.

## How it works

1. **A pool of clean clips per speaker** (`refs.extract_refs`, `styles`). Up to 12 of the
   speaker's lines of 2-10 s, the most typical first. Each must have only that speaker in it
   (purity ≥ 0.95), and the ASR must hear it again as its line's text, within CER 0.15. That is
   M8's rule for references: a clip holding a laugh or another voice makes the clone say it.
   - In a dialogue nearly every line has another speaker within 300 ms. Excluding those left
     the drama's pool with 3 clips out of 41, so such a line is now cut inside its own edges
     instead.
2. **The delivery of every line** (`dub.ops.prosody`), on the separated vocals:
   - pitch: median, in semitones (YIN, vectorised over frames);
   - pitch range: p90 - p10;
   - level of the active frames;
   - share of voiced frames (low when sobbing, whispering or breathy).
3. **A clip per line** (`prosody.pick`). The features are z-scored on the speaker's own lines,
   and the closest pool clip is chosen. It replaces the usual reference only when it is closer
   by 0.5 (z units), so neutral lines keep the usual voice.
4. **Take 0** of the line is cloned from that clip. Retakes, and the other engines a retry
   route names, use the usual reference: when a take fails the QA gate, the safe voice comes
   next.

`units.json` records each line's clip (`voice_ref`, the unit it was cut from).
`metrics.json` measures whether the dub follows the source from line to line, with each
speaker's lines centred on that speaker's mean:
- `prosody_f0_corr`, `prosody_f0_range_corr`, `prosody_energy_corr`: correlations with the
  source;
- `prosody_f0_spread`: the dub's pitch spread over the source's.

In `dub.yaml`: `voice.emotion_refs`, `voice.styles`, `voice.style_margin`.

## Measured

Same translations, A/B on the pod (`--param dub.voice.emotion_refs=false` for A).

| Drama (89 dubbed lines) | One reference per speaker | By delivery |
|---|---:|---:|
| Lines voiced from a delivery clip | 0 | 60 (pool: 28 clips) |
| Pitch follows the source, correlation | -0.10 | **0.46** |
| Pitch spread, dub / source | 0.26 | **0.81** |
| QA passed | 0.966 | **0.978** |
| Back-check CER p95 | 0.14 | 0.14 |
| Voice similarity / speaker's own, p50 (p10) | 0.89 (0.62) | 0.81 (0.59) |
| Lines over their slot (overflow) | 2 | 3 |

| Cartoon (9 dubbed lines with pitch) | One reference | By delivery |
|---|---:|---:|
| Pitch follows the source, correlation | 0.34 | 0.47 |
| Voice similarity p50 | 0.63 | 0.72 |

What it costs:
- The voice is a little less like the speaker's most typical clip (p50 0.89 → 0.81), well
  above the gate's 0.5 floor, and no line fails the gate for its voice.
- On the timing side, the 58 re-voiced lines of the drama gained 10 windows of 250 ms where
  the dub is silent while the actor speaks (44 → 54). They are mostly long, slow lines, voiced
  from slower clips. In the whole drama, the windows where dub and actor disagree went from 253
  to 244 (silent after the line ended 56 → 42, speaking over a silent actor 134 → 125).
- The pitch range and level do not follow the source better (correlations 0.22 → 0.07 and
  0.14 → 0.16). The pitch does.

## Two faults found on the way

- `_retranscribe_refs` and the audition rebuilt the reference bank without the pool: the first
  run voiced no line from its delivery. A test now holds the pool through them.
- **Worker model loads at once.** A call for OmniVoice that arrived before the background
  preload reached it loaded OmniVoice itself, while the preload thread loaded another model in
  the same process. Two threads importing transformers at once fail, because its lazy module
  is not thread-safe ("cannot import name 'AutoFeatureExtractor'"). The process stayed broken:
  every later call failed with "expected mat1 and mat2 to have the same dtype". It showed when
  many new voices arrived right after a restart. Now one model loads at a time in a worker
  process, and the transformers names the models use are imported before the threads start.

## Not done

- **A learned emotion embedding** (emotion2vec and the like) instead of four prosodic
  features. It would see anger vs joy at the same pitch.
- **The pool from more than the speaker's clean lines.** A minor character with no 2-10 s
  line keeps one voice.
- **Pitch range and level** still do not follow the source. The clip sets the mood; the
  engine's own sampling sets the rest.
