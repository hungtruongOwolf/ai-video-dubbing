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

## M1: Walking skeleton on a GPU (next)

- [ ] Worker SDK: an HTTP/JSON server per model family, Unix socket on the pod; executor `http` runtime
- [ ] Pod image + network volume layout (envs, weights pinned by commit, compile cache)
- [ ] `dub up/down`:
  - create the pod in the volume's datacenter, trying GPUs in order A40 → A6000 → L40S;
  - terminate the pod when idle;
  - upload the audio, download the results
- [ ] Real providers:
  - Mel-Band RoFormer separation;
  - faster-whisper large-v3, PhoWhisper (vi), word alignment;
  - OmniVoice with exact duration
- [ ] Rubberband time stretch on the pod
- [ ] `CLAUDE_CODE_OAUTH_TOKEN` on the pod
- [ ] Exit criteria:
  - one real en→vi video is watchable;
  - a rerun is 100% cache hits;
  - a 60-minute draft finishes in ≤ 12 minutes;
  - the report shows GPU minutes and $ per stage.

## M2: Edit loop

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
