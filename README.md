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
| **M2** Edit loop | Per-line edits, review page, partial re-render (one line redone in ~2 s) | **done** |
| **M3** Segmentation + timing | SaT sentence boundaries, VAD-snapped slots (onset error 10 ms), measured speech rates, subtitles per script and timed on the dub | **done** |
| **M4** Speakers + routing | Diarization (ECAPA), one cloned voice per speaker, QA gate with takes, forms-of-address checker, cast corrections, second TTS engine (VieNeu) | **done** |
| **M5** Throughput | Continuous batching on the GPU, LLM in parallel with TTS, 24 GB profile, many languages from one analysis: a 60-min video in 14 min (final) / 8 min (draft) | **done** |
| **M6** GPU lab | Models parked in CPU memory between jobs (kept), CUDA MPS (−6.5%, opt-in), community cloud (priced, not built) | **done** (optional) |
| **M7** Quality pass | Ideas from 11 open-source dubbing projects, each measured: separation at 44.1 kHz + autocast, silence compensation (LLM rewrites 140 → 25), per-line loudness, Vietnamese punctuation transfer (sentence F1 0.687 → 0.779 on 8 videos), worker restarts; voice audition measured, no gain | **done** |
| **M8** One pass | A video in, the final dub out, no manual fixes: who says each line read from the script (speaker changes that change voice 22% → 90% on a zh drama), burned-in subtitles read by OCR to correct the transcript, songs left original, clean voice references, short lines and accented takes voiced by VieNeu (short lines understood 35% → 64%, lines heard as Chinese 22% → 3.7%); on the drama QA 0.84 → 0.90, CER p95 0.30 → 0.14 | **done** |
| **M9** Faster runs, better lines | VieNeu loaded with the pod and workers serving while they load (cartoon job 175 s -> 71 s, drama voicing 227 s -> 95 s); speakers read 3 times and voted, on Fable (B-cubed F1 0.71 -> 0.78); laughs and cries keep the original; wordings measured, not predicted; drama QA passed 0.90 -> 0.99 | **done** |
| **M10** Expert review | Seven reviewers read the code: voicing 87.8 s -> 74.5 s on the same lines, several videos on one pod, laughs/cries/screams between lines kept without leaking the source language, stems that add up, prompts v2 (preferred ~55% by blind judges), short lines that fill their slots | **done** |
| **M11** Phrase timing | The dub pauses where the actor pauses: each take cut at its silences and re-timed to the actor's phrases, heard again before it is kept (drama: dub silent while the actor speaks 112 -> 63 windows; cartoon 15 -> 10) | **done** |
| **M12** Voices by delivery | Each line cloned from the speaker's clean clip closest to its own delivery (pitch, range, level, voicing): on the drama the dub's pitch follows the source (correlation -0.10 -> 0.46, spread 0.26 -> 0.81 of the source's), QA passed 0.966 -> 0.978 | **done** |
| **M13** Emotion from emotion2vec | emotion2vec+ (FunASR, own env) measures each line's emotion; a line's reference clip is picked by prosody and emotion together: on the drama the dub's emotion follows the source more (0.17 -> 0.24) at the same pitch agreement and QA | **done** |

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

- `interview.vi.mp4`: the original video stream (not re-encoded), the dubbed audio as its only audio track (`--keep-original-audio` adds the original as a second track; many players ignore the default flag and would play it), and two subtitle tracks;
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

### From a link

```bash
uv run dub get https://www.youtube.com/watch?v=...         # save the video in Downloads
uv run dub get                                             # a local page: paste links, Download
uv run dub run https://www.tiktok.com/@user/video/... --to vi --remote   # download, then dub
```

Downloads go through [yt-dlp](https://github.com/yt-dlp/yt-dlp), which knows over a
thousand sites (YouTube, TikTok, Facebook, X, Bilibili, Vimeo...). It picks H.264 first,
then the highest resolution, so every player can play the file and the dub. Some sites
only answer a browser that has visited them (Douyin, Instagram, private posts): pass your
browser's cookies with `--cookies-from-browser firefox`, or export a `cookies.txt` and pass
`--cookies FILE`. On Windows, Chrome and Edge encrypt their cookies so that other programs
cannot read them, so use Firefox or the export. Download only what you have the right to.

### Fix lines and re-render only what changed

```bash
uv run dub review video.mp4 --to vi --from en --remote     # review page at http://127.0.0.1:8765
```

Riskiest lines first; play the original and the dub, fix the text, ask for a new take, change the speaker, move, split or merge a line, then render. Only the touched lines are redone. Details: [docs/m2-edit-loop.md](docs/m2-edit-loop.md).

### Several speakers

Speakers are found automatically (`--speakers N` if you know how many): the voices are
clustered, then the LLM reads the script line by line and decides who says each line (a
question and its answer are two people even when the voices sound alike). Each one gets their
own cloned voice, and the translation uses the forms of address of their relationship
(Vietnamese "mẹ/con", "ông/cháu"...). To check who is who before anything is translated:

```bash
uv run dub run video.mp4 --to vi --from en --remote --profile final --confirm-cast
uv run dub edit video.mp4 --to vi cast S2 gender=female name=Linda
uv run dub edit video.mp4 --to vi cast S1>S2 self_term=mẹ other_term=con
uv run dub run video.mp4 --to vi --from en --remote --profile final
```

The review page has the same corrections in its cast panel. Details:
[docs/m4-speakers-routing.md](docs/m4-speakers-routing.md).

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
| `dub run VIDEO --to LANG[,LANG...] [--fake] [--use OP=PROVIDER] [--param PROVIDER.KEY=VALUE] [--profile draft\|final] [--speakers N] [--subs-only] [--confirm-cast]` | Dub one video, into one or several languages (the analysis runs once); `--confirm-cast` stops at the cast sheet to correct it |
| `dub get [URL ...] [--download-dir DIR] [--cookies-from-browser B] [--cookies FILE]` | Download videos from links; without a link, open a local page to paste them. `dub run URL` and `dub review URL` download first |
| `dub up` / `dub down` / `dub status` | Rent, terminate and inspect the GPU pod |
| `dub review VIDEO --to LANG [--remote]` | Review page: listen, fix lines, render, export |
| `dub edit VIDEO --to LANG ACTION ...` | Record an edit from the terminal (text, regen, speaker, nudge, split, merge, lock, undo, cast) |
| `dub bench seg VIDEO --golden FILE --remote` | Analyse a video (no dubbing) and score its sentence boundaries against the sentences really spoken |
| `dub bench speed VIDEO --to LANG --remote [--fresh]` | Dub a video and show where the time goes: stages, lanes, ops, GPU utilisation |
| `dub gc [--remote] [--older-than-hours N]` | Delete cache files no stored result points to; with `--older-than-hours`, also forget older audio results (transcripts and translations stay) |
| `dub doctor` | Show which providers can run on this machine and the route matrix |
| `dub schema` | Export the `/v1` data contracts as JSON Schema |
| `dub jobs` | List jobs in the workspace |

## Repository layout

```
configs/            what to run: dub.yaml, langpacks/, providers/ (one manifest per model), prompts/
src/dub/contracts/  /v1 data contracts between steps (pydantic -> JSON Schema)
src/dub/core/       thin core: config, router, cache-aware call context, executor, pipeline, trace
src/dub/store/      CAS (files by content hash), memo (results by call key), SQLite index
src/dub/ops/        pure CPU steps: ffmpeg, VAD, segmentation, timing fit, mixing, subtitles per script
src/dub/providers/  model adapters (fake_*, claude_cli, Whisper, RoFormer, OmniVoice, VieNeu, SaT, ECAPA)
src/dub/worker/     worker process: hosts the providers of one env over HTTP/JSON
src/dub/pod/        dubd, the pipeline server on the GPU pod (envs, jobs, uploads, idle watchdog)
src/dub/remote.py   laptop side of the pod: Runpod API, dub up/down/status, remote runs
src/dub/runner.py   one run of one video, local or remote (shared by dub run and the review page)
src/dub/edits.py    the human edit store (append-only JSONL on the laptop)
src/dub/review/     the review page (local web app)
src/dub/fetch/      dub get: videos from links (yt-dlp) and its local page
src/dub/bench/      measuring a job: segmentation F1 against a reference, speed and GPU use
experiments/        the measurements behind the defaults (E1, E3, E4, E6-E10, subtitle layout)
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
- [docs/m2-edit-loop.md](docs/m2-edit-loop.md): the review page, edits and partial re-rendering
- [docs/m3-segmentation-timing.md](docs/m3-segmentation-timing.md): sentence boundaries, slot snapping, speaking rates, subtitles per script; the experiments behind them
- [docs/m4-speakers-routing.md](docs/m4-speakers-routing.md): diarization, voices and the QA gate, forms of address, ASR per language, the second TTS engine
- [docs/m5-throughput.md](docs/m5-throughput.md): where the time of a 60-minute video goes and what made it faster
- [docs/m6-gpu-lab.md](docs/m6-gpu-lab.md): GPU experiments (CPU-memory offload, MPS, community cloud)
- [docs/m7-quality-pass.md](docs/m7-quality-pass.md): ideas from other dubbing projects, measured (separation, silence, loudness, voice audition, segmentation trials)
- [docs/m8-one-pass.md](docs/m8-one-pass.md): one pass without manual fixes: who says each line, burned-in subtitles, Chinese left in a dub, short lines, voice references
- [docs/m9-faster-better.md](docs/m9-faster-better.md): where the time went (VieNeu loaded on first use, workers waiting for each other), speakers voted over 3 readings on Fable, the translation model judged blind, laughs kept original
- [docs/m10-expert-review.md](docs/m10-expert-review.md): the seven-reviewer audit and what it measured: the GPU waiting on orchestration, non-verbal sounds, translation v2, several videos at once
- [experiments/README.md](experiments/README.md): how to re-run the measurements
- [docs/claude-subscription.md](docs/claude-subscription.md): translating with `claude -p`
- [docs/roadmap.md](docs/roadmap.md): milestones and exit criteria

## License

The code is MIT. Models used by later milestones have their own licenses, some of them non-commercial. This is a non-commercial learning project.
