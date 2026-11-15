# M9: faster runs, better lines

Goal (2026-11-15): make the whole process faster without losing quality, and fix what kept
going wrong in the dubs (wrong speaker, clipped words, laughs dubbed as words). Test material:

- the 5-minute zh drama of M8, with its hand labels of who says each of 177 subtitle cues;
- a 61-second 3D cartoon (a dragon, a boy, an old man) with animated burned-in subtitles.

Every number below comes from a run on the pod (RTX PRO 4500, `final` profile, `--fresh`
unless noted) or from a script in `experiments/`.

## Where the time went

`dub bench speed` on the jobs of M8 showed the GPU busy 4% (cartoon) and 10% (drama) of the
job, while the TTS lane was "busy" 82-96% of it.

| Finding | Measured | Fix |
|---|---|---|
| VieNeu (short lines, accent retries) loaded on first use | first VieNeu calls of a job took 130-196 s; later ones 0.2-0.6 s | loaded with the pod; the warm-up goes through every path a line takes (watermark, enrolling a voice and its denoiser) |
| dubd held a job until every worker had loaded | 115 s of "waiting for the model workers" on a cold pod | a job starts at once; a call to a worker still loading waits for that worker (`DUB_WORKERS_PENDING`) |
| A worker served only after loading all its models | diarization (ECAPA) waited for OmniVoice in the same worker | workers serve while they load, in the order a job first needs the models; the voice-hint model is preloaded |
| The envs were checked one after the other | the tts worker started last | all envs at once; real installs still one at a time (`flock`) |
| The accent check transcribed every take to read its language | 128 Whisper transcriptions, 732 s of the ASR worker on the drama | `detect_language` only: same answer on 40 of 40 takes, 89 ms against 460 ms (`experiments/lid_check.py`) |
| The laptop read the subtitles (OCR) before renting the pod | 30-60 s | the pod is rented in a thread while the video is read |

| Fresh run, warm pod | Before | After |
|---|---:|---:|
| Cartoon (1:01): whole job | 175-234 s | **71 s** |
| Cartoon: voicing | 136 s | **17 s** |
| Drama (5:14): voicing | 227 s | **95 s** |
| GPU busy, cartoon / drama | 4% / 10% | 19% / 47-55% |

From a cold pod (rented by `dub run`, laptop wall clock until the dub is downloaded), the
cartoon took about 8 minutes before (245 s of boot, then the job). Five cold runs of the
final code took 223, 266, 280, 261 and 243 s. The floor is reading the model weights from
the network volume: RoFormer takes about 60 s to load, the tts worker about 2 minutes.
Spread over these runs, the start-up changes did not move the cold time measurably. Only
the VieNeu fix did. A faster cold start needs the weights on faster storage (baked into
the image, or on a local disk); that is not done here.

## Quality

### Who says each line

`text.speakers` now reads the script 3 times (concurrently) and each line goes to the
character most readings give it. Characters are matched across readings by the lines they
share. A line the readings disagree on is marked as a guess and flagged for review.

The model was then compared, scored with `experiments/phrase_speakers.py` against the hand
labels:

| Attribution | B-cubed F1 | Speaker changes: precision / recall | Readings agree |
|---|---:|---:|---:|
| Sonnet, 1 reading (M8) | 0.711 | 0.91 / 0.82 | - |
| Sonnet, 3 readings | 0.756 | 0.84 / 0.84 | 71% |
| Opus, 3 readings | 0.776 | 0.88 / 0.88 | 93% |
| Fable, 3 readings | 0.775 | 0.92 / 0.88 | 94% |
| **Fable for speakers, Sonnet for translation (new default)** | **0.782** | **0.92 / 0.88** | 95% |

The flags point at the errors. In the Sonnet 3-reading run:
- 41% of the flagged lines carry the wrong voice, against 9% of the others;
- 30 of its 39 wrong lines are flagged.

The cartoon:
- with Sonnet, the attribution never found the boy;
- with Fable, all 3 readings agree on 3 people, the boy included ("Nailong's friend who
  warns him").

### Which model translates

`experiments/judge_translations.py` collected the 62 lines that the Sonnet, Opus and Fable
runs voiced differently, from the same source line. Three judges (Opus, Fable, Sonnet) each
ranked those lines blind, on meaning, natural speech and forms of address.

The result was even:
- mean rank 1.94-2.08 for every model;
- every pair split 43-52%.

Sonnet also wrote the context and the translation in 52 s, against Opus's 168 s. So
translation stays on Sonnet, and only `text.speakers` moves to Fable
(`op_params` in the provider manifest).

### Lines

| Problem seen | Cause | Fix |
|---|---|---|
| 哈哈 dubbed "Ha", overrunning a 20 ms slot | laughter translated like words | a line made only of laughter, cries or fillers (langpack `interjections`: 哈哈, 啊, uh-huh, はは...) keeps its original audio |
| "Không hứng" for "Không hứng thú" | the short-line boost, measured for OmniVoice, made VieNeu's full line look too long | a short wording picked on a prediction is voiced again in full, measured, and kept when it fits; a line whose every take fails the gate tries the other wordings |
| VieNeu said "không hơn" in 9 takes of 9 | the engine, not the wording | a line whose takes all fail the gate gets one take from OmniVoice (`qa-retry` route) |
| A short reply missing from the transcript (没兴趣) | OCR read a fraction of short lines; the ASR missed it | OCR cut to the text; a whole missed cue becomes its own word (see `docs/m8-one-pass.md`) |
| Two jobs at once crashed a model ("weight is on cpu") | one job unloaded a model while the other ran it | a worker holds its lock for the whole call |
| `dub run --to vi,es` without a pod | each language rented its own pod | one lock around renting |

Drama, before and after (M8's final run, then the final code of this round):

| | M8 | Now |
|---|---:|---:|
| Takes passing the QA gate | 0.899 | **0.989** |
| Back-check CER p95 | 0.138 | **0.12** |
| Lines worth a listen | 14 of 86 | **11 of 96** |
| Forms of address flagged | 4 | 2 |

The cartoon:
- QA passed 1.0;
- CER p95 0.11;
- no overrun;
- every line present.

## Not done

- **Faster cold start:** model weights on faster storage than the network volume (above).
- **The network volume is nearly full** (50 GB; two jobs at once hit the quota).
  - Option: 10 GB more costs about $0.70 a month.
  - Until then, run one job at a time and `dub gc --remote` after heavy runs.
- **Who speaks from the picture:** the script alone cannot always tell (the boy and the
  dragon in the first cartoon runs); the video frames are not used.
- **Listening test:** still no human ears on these numbers.
