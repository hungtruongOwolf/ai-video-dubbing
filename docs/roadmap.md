# Roadmap

One person, part time (about 12–15 hours a week). Every milestone ends with:

- a command that runs;
- a short note in `docs/`;
- green CI.

Reference hardware from M1:

- 1× A40 48 GB on RunPod Secure Cloud ($0.59/h), with a network volume;
- every model stays resident in VRAM;
- the pod is terminated when idle.

## M0: Foundation (done)

- [x] `/v1` data contracts (pydantic → JSON Schema, `dub schema`)
- [x] CAS + memo + SQLite index; atomic writes; deterministic keys
- [x] `Ctx.call` / `Ctx.local`: routing + memoisation + retry/fallback on classified errors
- [x] Router: CLI overrides → rules → langpack chain → defaults, capability filter, route trace
- [x] Provider manifests; `inproc` runtime with per-provider concurrency limits
- [x] Fake providers for every op; `dub demo` test video with a replayable transcript
- [x] Baseline segmentation (hard boundaries, abbreviations, DP splitting, tiny-unit merge)
- [x] Timing ladder: variant choice, bounded-ripple borrowing, safe ratio, rewrite hook, stretch
- [x] Mix + loudness normalisation; SRT with balanced lines and DP cue splitting
- [x] Laptop side: audio-only extraction (bit-exact), mux with `-c:v copy`, 2 audio + 2 subtitle tracks
- [x] Cast sheet in the context pass; per-window cast rows in the translation key
- [x] `claude_cli` provider (subscription), tested with a fake CLI and once for real
- [x] Report, Perfetto trace, `dub doctor`, CI (ruff + pytest + end-to-end with ffmpeg)

**Exit criteria met:**

- `dub run demo/interview.mp4 --to vi --fake` produces a valid MP4 on the laptop.
- A rerun is 100% cache hits.
- CI finishes in under 2 minutes.

**Changed from the original plan:** the pod-side items (worker HTTP SDK, RunPod template, `dub up/down`) moved to M1. They can only be verified against a real pod.

## M1: Walking skeleton on a GPU (done)

- [x] Worker processes: one per model family (env), HTTP/JSON, models preloaded and resident
- [x] Executor `http` runtime; the router only offers providers whose worker is running
- [x] `dubd` on the pod:
  - installs worker envs on the network volume (cached);
  - supervises the workers;
  - chunked uploads, jobs with live logs, downloads;
  - idle self-termination
- [x] `pod/bootstrap.sh`: uv, static ffmpeg with rubberband, the repo at the laptop's exact commit
- [x] `dub up` / `dub down` / `dub status`:
  - GPU list tried in order;
  - an in-place code update of a running pod (no new rental);
  - `dub run --remote`
- [x] Real providers:
  - Mel-Band RoFormer separation;
  - Whisper large-v3 and PhoWhisper (faster-whisper, word timestamps);
  - OmniVoice with exact duration;
  - one voice reference clip per speaker
- [x] Claude on the pod through the subscription (`claude_code_oauth_token` Runpod secret)
- [x] Rubberband time stretch on the pod; AAC output so the download is small

**Exit criteria:**

| Criterion | Result |
|---|---|
| A real en→vi video is watchable | yes. Whisper transcribed the source almost perfectly, the timing fit needed no flags, and a Whisper back-check of the Vietnamese dub recovered nearly every line. |
| A rerun is 100% cache hits | yes: 39/39, 17 s end to end including upload, download and mux |
| 60-minute draft ≤ 12 min | not measured yet. From the stage timings of the test clip, the estimate on an RTX 4090 is ~18–20 min: stages still run one after another and TTS runs one line at a time. Batching and running the LLM alongside TTS are M5. |
| Report shows GPU minutes and $ per stage | partly: seconds per op (including queue wait). Dollars per stage come with `dub bench` (M5). |

Measured on a 65 s two-voice English clip (RTX 4090, pod warm):

| Stage | Time |
|---|---|
| separation (Mel-Band RoFormer) | 5.3 s |
| ASR (Whisper large-v3, in parallel with separation) | 5.1 s |
| context pass (Claude, cast sheet) | 17 s |
| translation, 2 windows in parallel (Claude) | 17 s |
| TTS 13 lines + fit + stretch | ~10 s wall |
| mix + AAC | 5 s |
| **whole job** | **~50 s** |

First boot on an empty volume, from pod creation to all workers ready: about 5 minutes. A restart with everything cached: about 1.5 minutes.

**Changed from the original plan:**
- word alignment with wav2vec2 moved to M3 (faster-whisper word timestamps are used for now);
- multi-speaker diarization stays in M4: `single_speaker` gives one cloned voice per video, and Claude's cast sheet flags this itself;
- the A40 is often out of stock. The volume moved to EU-RO-1, which has more GPU types, and the M1 model set fits a 24 GB card.

## M2: Edit loop (next)

- [ ] Field revisions + locks + `edits.jsonl`
- [ ] Split/merge units
- [ ] Dirty-set re-render: only the edited line and its neighbours
- [ ] Review page:
  - riskiest lines first;
  - listen ±5 s;
  - edit, regenerate, pick a take, change the speaker
- [ ] Offline edits while the pod is off (`dub edit export/import`)
- [ ] Exit criteria: editing one line re-renders in a median of ≤ 2 s.

## M3: Segmentation + timing (priorities 1 and 2)

- [ ] SaT boundary probabilities in the DP (experiment E4)
- [ ] Measured speech-rate model per voice (E7)
- [ ] Engine safe ratios measured (E3)
- [ ] Dub-side alignment → subtitle cues follow the new voice
- [ ] Subtitle layout per script:
  - Thai word breaking;
  - CJK rules;
  - right-to-left text
- [ ] Exit criteria:
  - no unit longer than 15 s;
  - onset error p95 ≤ 150 ms;
  - speed-up p95 ≤ 1.15

## M4: Speakers + routing + second engine

- [ ] Diarization (pyannote → Nemotron if E8 confirms)
- [ ] Unit purity check
- [ ] Reference voice bank (anchor + local clips, quality gate)
- [ ] Voice age/gender model as cast-sheet evidence
- [ ] `--confirm-cast` mode
- [ ] Rule-based forms-of-address checker per language
- [ ] Whole-script address consistency pass (E10)
- [ ] QA gate: ASR back-check, speaker similarity, truncation; deterministic k-best takes
- [ ] Second TTS engine with continuous batching
- [ ] Blind A/B experiments (`experiments:`)
- [ ] Exit criteria:
  - no speaker swaps;
  - forms of address correct on ≥ 95% of lines in a 4-person video;
  - switching the vi TTS takes ≤ 3 lines of config

## M5: Throughput

- [ ] Batching across units
- [ ] LLM in parallel with TTS
- [ ] `dub bench`
- [ ] 24 GB GPU profile (two phases)
- [ ] One source → many target languages (Phase A runs once)
- [ ] Exit criteria:
  - a 60-minute final finishes in ≤ 25 minutes;
  - the GPU is busy ≥ 70% of billed time

## M6: GPU lab (optional, never blocks the product)

CUDA graphs vs continuous batching, MPS, pinned-RAM model swapping, local LLM, community-cloud profile. Each experiment = a short note + one line in the trend table.
