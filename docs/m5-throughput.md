# M5: throughput

How long a real hour of video takes, where the time goes, and what made it shorter. All
numbers are from the pod (RTX 4090, 24 GB, 48 vCPUs) dubbing the same video, measured with
`dub bench speed --fresh` (every step recomputed).

**Test video:** the 2013 State of the Union address (Wikimedia Commons, public domain),
60.1 minutes, English, one main speaker with long applause breaks: 6907 words, 404
sentences, 418 dub units. Target: Vietnamese.

## Measuring

- `dub bench speed VIDEO --to vi --remote [--profile final] [--fresh] [--json F]` dubs the
  video and splits the wall time into stages (the `[k/7]` marks on the trace), lanes ("gpu ·
  tts", "llm", "cpu"...) and ops; `dub bench speed JOB_DIR` re-reads a finished job.
- dubd samples `nvidia-smi` every 0.5 s during each job (`gpu.csv` in the job folder); the
  trace records its wall clock, so each stage gets its GPU utilisation.
- `--fresh` ignores everything stored before the job; what the job computes itself is
  reused within the job (voicing ahead depends on it).

## Where the time went (draft, before M5)

| Stage | Wall | GPU |
|---|---|---|
| 1 decode | 6 s | 0% |
| 2 separate ∥ transcribe | 179 s | 92% |
| 3 who speaks when | 9 s | 24% |
| 4 cast sheet (LLM) | 14 s | 0% |
| 5 translate (35 windows) | 139 s | **0%** |
| 6 voice and fit 418 lines | 219 s | 43% |
| 7 mix, back-check, master, subtitles | 236 s | 64% |
| **total** | **801 s (13.4 min)** | **51%** |

Three things stood out: the GPU idled through the whole translation; TTS ran one line per
call, two calls at a time; and the end of the job waited on the back-check of all 418 lines
(started only after the last line was placed), a dynamic loudness normaliser and the
subtitle layout. It also failed 18 back-checks: CUDA out of memory, four workers sharing
24 GB.

## What changed

| Change | Where | Effect |
|---|---|---|
| Continuous batching in the workers: queued calls with the same provider, op and params run as one `run_batch` | `dub/worker/server.py`; OmniVoice, Whisper back-check, ECAPA | TTS 0.84 → 0.22 s per line (12 lines at once) |
| OmniVoice batches with one seeded sampling generator per line | `providers/tts_omnivoice.py` | a line's randomness is its own (see "Reproducibility") |
| Back-checks batched: clips joined with 0.5 s gaps, one `clip_timestamps` entry each, words split back by time | `providers/asr_whisper.py` | batches of 16 |
| Voicing ahead: a translated window's lines are voiced while the others are translated | `core/pipeline.py` | translation no longer leaves the GPU idle |
| Back-checks start as each line is placed | `core/pipeline.py` | the post-hoc QA overlaps voicing |
| 24 GB profile: phase-A models unloaded after phase A; workers hand cached GPU memory back; out-of-memory retried | `pod/`, `worker/`, `core/ctx.py` | 0 failures (was 18) |
| Master: measured gain + true-peak limiter + AAC fast coder | `ops/mix.py` | 114 → 43 s |
| Subtitle layout: cached widths, linear fit test in the cue split | `ops/subs.py` | 12 → 2.5 s for 418 synthetic lines (laptop); 14 s on the pod before |
| LLM calls in flight 4 → 8 | `claude_cli.yaml` | translation 139 → 78 s |
| Final: lines too fast for an exact-duration engine are rewritten before they are voiced | `core/pipeline.py` | one synthesis instead of two |
| Single flight: identical calls in flight are computed once | `core/ctx.py` | shared analysis across target languages |

## Results (60.1 min, en → vi, RTX 4090)

| Run | Wall | Speed | GPU utilisation |
|---|---|---|---|
| draft, before M5 | 801 s (13.4 min) | 4.5× real time | 51% |
| draft, batching (voicing ahead redone by mistake: 836 syntheses) | 637 s | 5.7× | 77% |
| draft, M5 | **470 s (7.8 min)** | 7.7× | 73% |
| final, M5 | **844 s (14.1 min)** | 4.3× | **81%** |

Draft, M5, by stage: separate ∥ transcribe 179 s · who speaks 9 s · cast sheet 14 s ·
translate 141 s (GPU 79%: lines are voiced meanwhile) · voice and fit 52 s · mix and
finish 68 s.

Final, M5, by stage: separate ∥ transcribe 179 s · who speaks 10 s · cast sheet 15 s ·
translate 78 s (GPU 75%) · voice, QA and fit 495 s (GPU 93%; 705 syntheses for 418 lines:
second and third takes, 136 rewrites) · mix and finish 61 s. No call failed.

## Exit criteria

| Criterion | Result |
|---|---|
| A 60-minute final finishes in ≤ 25 minutes | **14.1 min** (draft 7.8 min) |
| The GPU is busy ≥ 70% of billed time | **81% of the job**. Counting a warm pod start (171 s) as well: 68%; a cold first boot adds ~5 min. Billing also runs through the idle timeout (10 min) unless `dub down` follows the job, so batches of videos or languages per pod session are what keep the whole bill busy. |

## One source, many languages

`dub run VIDEO --to vi,es` submits one job per language at the same time. On a fresh 3-minute
clip, the Vietnamese job computed the analysis (decode, separation, ASR, diarization, VAD,
sentence boundaries, segmentation, voice references) and the Spanish job reused every one of
those calls (single flight), computing only its own cast sheet, translation, voices and mix.
Both finished in ~113 s.

## Reproducibility

Batched OmniVoice takes are not bit-identical to the same line synthesised alone: batched
kernels round differently and the diffusion sampler amplifies it (measured: 3 of 4 lines came
out as different takes; one was identical). Each line still samples from its own generator
seeded with its own seed, so neighbours' seeds never change it. Every take is stored, so
reruns, resumes and edits reuse it; only a line recomputed after its cache entry is gone can
change, to an equally valid take. The architecture's rule ("the scheduler changes when, not
what") now carries this one measured exception, for 3.9× TTS throughput.

## Phase A: what was tried and not kept

Phase A (separation ∥ ASR, 179 s) is now the largest fixed cost of a draft.

**Batched Whisper for the source transcript.** faster-whisper's BatchedInferencePipeline
(VAD chunks, batch 16) transcribes the hour in 35 s against 85 s sequentially, on an idle GPU
(6802 words against 6907). But sentence segmentation, priority 1, gets worse
(`real_videos.py run --param whisper_large_v3.batch_size=16 --tag m5batched`, F1 at ±1 word):

| Language | Sequential | Batched |
|---|---|---|
| en (3 videos) | 0.843 | 0.817 |
| es (2) | 0.833 | 0.778 |
| vi (2, PhoWhisper) | 0.697 | 0.661 |
| zh (2) | 0.801 | 0.847 |
| ja (2) | 0.833 | 0.833 |
| **all 11** | **0.805** | 0.790 |

Kept sequential (`batch_size: 0`); batching stays for back-checks, which never feed the
segmentation. In the pipeline the ASR runs next to the separation anyway, and the separation
(155 s alone) sets the stage's length.

**Separation batch size (kept).** The Mel-Band RoFormer runs its chunks one per forward pass
by default. With `batch_size: 8` the hour separates in 117.5 s instead of 155.4 s (4: 140.1 s),
with bit-identical vocals (maximum difference 0.0 over a 60 s excerpt).
