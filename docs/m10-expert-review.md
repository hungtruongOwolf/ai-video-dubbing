# M10: the expert review, and what came of it

On 2026-11-15, seven read-only reviewers went through the code, the docs and the two test
jobs: an engineering manager, a speech researcher, a GPU engineer, a CUDA engineer, a speech
quality engineer, a Chinese to Vietnamese dubbing translator and an audio post-production
engineer. The two test jobs are the 5:14 zh drama and the 61 s cartoon. This page lists what
was done with their findings and what it measured on the pod (RTX PRO 4500, `final` profile).

## Speed

The findings agreed: the GPU waited on the orchestration, not on the models.
- The default thread pool (32 threads) was full for much of the voicing.
- The three QA calls of a take ran one after the other.
- A 10 ms speaker embedding queued behind OmniVoice batches, and back-checks behind Whisper.
- VieNeu ran its lines one at a time.
- Phase-A models were freed on a 32 GB card that peaks at 16 GB.

Changes:
- QA calls at once;
- cheap calls served first;
- a 256-thread pool per job;
- VieNeu batches per voice;
- back-check (PhoWhisper) and speaker embedding (ECAPA) in their own worker processes
  (manifest `worker`);
- language ID by detection only;
- phase-A models freed only below 30 GB;
- every back-check on one decoding path;
- kernel caches on the volume.

The first fresh runs looked slower, not faster. A fresh run translates again, and the number of
hard lines, with their chains of retakes, changes from one run to the next. Compare runs on
the same cached translations, with only the voicing redone (a dummy provider param):

| Voicing of the drama, same translations | Seconds | Syntheses | Back-checks computed |
|---|---:|---:|---:|
| Code before M10 | 87.8 | 242 | 35 (the rest cached) |
| 24 lines in flight, retakes one after another (**default**) | **74.5** | 243 | 75 |
| 24 in flight, retakes at once | 77.6 | 258 | 84 |
| 48 in flight, retakes one after another | 79.2 | 235 | 86 |
| 48 in flight, retakes at once | 93.2 | 251 | 117 |

Two ideas did not hold up:
- Voicing the other takes at once when take 0 fails added work and lengthened the queues.
- 48 lines in flight did no better than 24.

The kept defaults are about 15% faster than before while computing twice as many back-checks.
The voicing tail is the retake chain of one or two lines the back-check cannot hear right (see
"Names" below).

**Several videos on one pod** (`dub run A B --remote`): dubd lets one job analyse at a time
(the VRAM peaks) while the others translate or voice. The cartoon and the drama together took
323 s on the laptop's clock, against 379 s of job time one after the other. The GPU works for
one job while another waits for the LLM.

**Observability:**
- `gpu.csv` now records power and SM clock. The drama's voicing drew 135 W on average,
  against a 203 W peak.
- Each job saves its workers' batch-size histogram (`batches.json`).

**Correction to M9.** M9 called the cold-start floor "reading the weights from the network
volume". That is wrong: Whisper's 3.6 GB loads in 9.7 s (M6). The floor is the imports, i.e.
thousands of small files of torch and transformers read from the volume, and the workers
starting one after another. Envs are now checked in parallel. Putting the envs on the container
disk is not done.

## Laughs, cries and screams

The audio engineer answered the question directly: no, most of them were lost.
- The separator puts every sound a person makes in the vocals stem, and the dub replaces that
  stem.
- Only lines the ASR transcribed could be kept in the original audio, and only within their
  slot. The cartoon lost a 3.3 s laugh and the last 1.8 s of a scream.

Now (`dub.ops.nonverbal`):
- Vocal sound with no line and no subtitle over it is passed to the source-language ASR.
- If it hears no words, or only laughter and cries (the langpack `interjections`), the sound is
  kept in the original audio.
- Words keep it out, since they would leak the source language (an answer the ASR missed,
  你瘦了).

| | Vocal gaps kept | Seconds kept | Kept out (words heard) |
|---|---:|---:|---:|
| Cartoon | 11-13 of 12-13 | 14.8-22.9 | 0-1 |
| Drama | 19-20 of 25-26 | 23.3-24.4 | 6 |

Also:
- **Stems that add up.** The separator normalises each stem, so the bed came out ~2 dB low.
  `audio.rebalance` fits a, b in mix ≈ a·vocals + b·bed (least squares) and rescales both
  stems; the test gets them back within -40 dB.
- **Kept originals** keep their stereo image and fade with 40 ms equal-power ramps.
- **Lines that are only a cry** come back empty from translation and get no subtitle.

## Translation

Prompts v2:
- translate meaning (idioms, dialect: 切 for 去);
- restore a misheard source word, with a note (买你 among things to buy: 买米);
- never write laughter out;
- avoid wordings a listener mishears ("con ngươi", "năm mười hai lạng");
- polite register for servants;
- `short` keeps numbers, names and negation; `long` never pads.

The context sheet adds aliases, motifs and a name policy: Hán-Việt for a period drama, a known
character by the name its audience uses.

Two blind judges ranked the 69 lines that v1 and v2 worded differently. Both preferred v2:
- Opus: 38 vs 30;
- Fable: 38 vs 31;
- about 55% of the lines either way, mean rank 1.44 vs 1.56.

**Names.** v2 keeps the names people call each other ("A Duyên à"), which the address fix loop
used to delete. PhoWhisper then mishears them: "ở mãn", "hai đậu", "ây tuyên", and "lão già"
for "lão gia". All 4 QA failures of the v2 run were name lines; the other lines failed none.
QA passed fell from 0.977-0.989 to 0.956 for that reason, not because the dub got worse. A
back-check that knows the cast's names is not done.

**Variants.**
- The target wording now comes first: it fits up to the 1.10 speed-up.
- The long wording is used only when the target would leave a quarter of the slot silent.
- A short wording under 70% of the target is not used while the target nearly fits.

**Short lines.** Short Vietnamese lines go to VieNeu only in slots up to 1.6 s. VieNeu has no
length control and left longer slots half silent. Lines filling their slot outside 0.7-1.2
went from 28 to 15 of 90 (p10 0.60 → 0.70, p90 1.28 → 1.08).

## Fixes

- **Address checker:**
  - a listener with no relation (a stranger) is not an error;
  - a name is a valid vocative.
- **Risk list:**
  - kept originals are no longer listed (two baby cries topped it);
  - a failed QA gate adds 3;
  - underfilled slots are flagged;
  - slowed lines are labelled "drawn out".
- **Voice gate.** Self-similarity comes from lines ≥ 1.2 s, with the job median for speakers
  that have fewer than 2.
- **Speaker attribution failures** are a warning, not a silent fallback.
- **Worker lock** (M9) and two flaky tests:
  - several videos wrote `sources.json` at once;
  - the download page replied before reading a POST body.

## Not done

- OmniVoice's FlashInfer path (upstream README: 2.1-2.6x), which needs sm_120 kernels checked
  and the per-line seeding rewritten.
- Qwen3-ASR or FireRedASR2 for Chinese sources (Whisper large-v3 trails them 2-3x on reported
  benchmarks).
- Reference clips chosen per line by emotion: every line is cloned from one clip, and in the
  drama that clip was a grieving line. Line-to-line pitch spread in the dub is 1/3 to 1/6 of
  the source's.
- A speaker embedding that does not confuse language with identity (ERes2NetV2).
- Envs on the container disk; splitting `pipeline.py`.
- A listening page for blind A/B.
