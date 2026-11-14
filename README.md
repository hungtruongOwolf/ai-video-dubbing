# ai-video-dubbing

A multilingual AI video dubbing pipeline, built step by step as a personal learning project.

```
video file ─► audio track ─► separate speech / background ─► speech-to-text ─► who speaks when
          ─► sentence segmentation ─► cast sheet (gender, age, relations, forms of address)
          ─► translation (3 length variants) ─► voice cloning TTS ─► timing fit ─► mix + subtitles
          ─► muxed back into the original video
```

Priorities, in order: **(1) sentence segmentation, (2) timing sync, (3) voice quality.**
Any language pair, in any direction.

## Status

| Milestone | What | State |
|---|---|---|
| **M0** Foundation | Data contracts, content-addressed cache, routing, fake providers, full pipeline on a laptop, Claude translation, CI | **done** |
| **M1** Walking skeleton | Real models on a Runpod GPU (Mel-Band RoFormer, Whisper/PhoWhisper, OmniVoice, Claude), `dub up/down/status`, `dub run --remote` | **done** |
| M2 Edit loop | Per-line edits, review page, partial re-render | next |
| M3 Segmentation + timing | Model-based boundaries (SaT), measured speech rates, subtitle layout per script | planned |
| M4 Speakers + routing | Diarization, voice references, cast sheet checker, second TTS engine | planned |
| M5 Throughput | Batching across units, parallel LLM + TTS, multi-language fan-out | planned |
| M6 GPU lab | Optional performance experiments | optional |

Details: [docs/roadmap.md](docs/roadmap.md).

## Quick start (laptop, no GPU, no cost)

Requirements: [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/) on `PATH`, Git.

```bash
git clone https://github.com/hungtruongOwolf/ai-video-dubbing.git
cd ai-video-dubbing
uv sync

uv run dub demo                                         # builds demo/interview.mp4 (+ transcript)
uv run dub run demo/interview.mp4 --to vi --fake        # dub it with fake models
```

Output in `demo/interview.dub/`:

- `interview.vi.mp4`: the original video stream (not re-encoded), a dubbed audio track, the original audio as a second track, and two subtitle tracks;
- `interview.vi.srt`, `interview.en.srt`.

The job folder (`dub-workspace/jobs/<job>/`) holds:

- `report.md`: calls, cache hits, routing decisions, timing flags, cast sheet;
- `trace.json`: a timeline you can open at <https://ui.perfetto.dev>;
- `units.json`: every unit with its translation and placement;
- `cast.json`, `routes.jsonl`.

Run the same command again. Every step is served from the cache (`calls: 37, cache hits: 37`).

### On a GPU (Runpod)

```bash
uv run dub up                                              # rent the pod, load the models
uv run dub run video.mp4 --to vi --from en --remote        # real separation, ASR, translation, TTS
uv run dub down                                            # or let it idle out after 10 minutes
```

One-time setup (API key, network volume, Claude token): [docs/m1-gpu-pod.md](docs/m1-gpu-pod.md).

### Real translation with your Claude plan

If the [Claude Code](https://docs.claude.com/en/docs/claude-code) CLI is installed and logged in:

```bash
uv run dub run demo/interview.mp4 --to vi --fake --use "mt.*=claude_cli"
```

Speech is still faked, but the context pass (summary, glossary, cast sheet) and the translation are real. See [docs/claude-subscription.md](docs/claude-subscription.md).

## Commands

| Command | Does |
|---|---|
| `dub demo` | Build a 1-minute test interview video and its transcript |
| `dub run VIDEO --to LANG [--fake] [--use OP=PROVIDER] [--profile draft\|final] [--subs-only]` | Dub one video |
| `dub up` / `dub down` / `dub status` | Rent, terminate and inspect the GPU pod |
| `dub doctor` | Show which providers can run on this machine and the route matrix |
| `dub schema` | Export the `/v1` data contracts as JSON Schema |
| `dub jobs` | List jobs in the workspace |

## Repository layout

```
configs/            what to run: dub.yaml, langpacks/, providers/ (one manifest per model), prompts/
src/dub/contracts/  /v1 data contracts between steps (pydantic -> JSON Schema)
src/dub/core/       thin core: config, router, cache-aware call context, executor, pipeline, trace
src/dub/store/      CAS (files by content hash), memo (results by call key), SQLite index
src/dub/ops/        pure CPU steps: ffmpeg, segmentation, timing fit, mixing, subtitles
src/dub/providers/  model adapters (fake_*, claude_cli, Whisper, RoFormer, OmniVoice)
src/dub/worker/     worker process: hosts the providers of one env over HTTP/JSON
src/dub/pod/        dubd, the pipeline server on the GPU pod (envs, jobs, uploads, idle watchdog)
src/dub/remote.py   laptop side of the pod: Runpod API, dub up/down/status, remote runs
pod/                pod bootstrap script and per-env requirement files
src/dub/laptop.py   laptop side: extract the audio track, mux the result back
tests/              unit tests, end-to-end test, architecture fitness tests
docs/               architecture, configuration, roadmap, milestone notes
```

## Documentation

- [docs/architecture.md](docs/architecture.md): how the system is put together and why
- [docs/configuration.md](docs/configuration.md): adding a language, a model or a routing rule
- [docs/m0-foundation.md](docs/m0-foundation.md): what M0 contains and a guided tour
- [docs/m1-gpu-pod.md](docs/m1-gpu-pod.md): running the real models on a Runpod GPU
- [docs/claude-subscription.md](docs/claude-subscription.md): translating with `claude -p`
- [docs/roadmap.md](docs/roadmap.md): milestones and exit criteria

## License

The code is MIT. Models used by later milestones have their own licenses, some of them non-commercial. This is a non-commercial learning project.
