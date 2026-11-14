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
| Subtitle layout: cached widths, linear fit test in the cue split | `ops/subs.py` | 12 → 2.5 s for 418 lines |
| LLM calls in flight 4 → 8 | `claude_cli.yaml` | translation 139 → 78 s |
| Final: lines too fast for an exact-duration engine are rewritten before they are voiced | `core/pipeline.py` | one synthesis instead of two |
| Single flight: identical calls in flight are computed once | `core/ctx.py` | shared analysis across target languages |

## Results (60.1 min, en → vi, RTX 4090)

| Run | Wall | Speed | GPU utilisation |
|---|---|---|---|
| draft, before M5 | 801 s (13.4 min) | 4.5× real time | 51% |
| draft, batching (voicing ahead redone by mistake: 836 syntheses) | 637 s | 5.7× | 77% |
| draft, M5 | **470 s (7.8 min)** | 7.7× | 73% |
| final, M5 | FINAL_PLACEHOLDER |

Draft, M5, by stage: separate ∥ transcribe 179 s · who speaks 9 s · cast sheet 14 s ·
translate 141 s (GPU 79%: lines are voiced meanwhile) · voice and fit 52 s · mix and
finish 68 s.

RESULTS_PLACEHOLDER
