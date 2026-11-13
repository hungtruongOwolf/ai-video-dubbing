# M0: Foundation, a guided tour

M0 builds the skeleton the real models will plug into. All of it runs on a laptop, without a GPU, in about 15 seconds per minute of video.

## 1. Run it

```bash
uv sync
uv run dub demo                                   # demo/interview.mp4 + demo/interview.words.json
uv run dub run demo/interview.mp4 --to vi --fake
```

What you see:

```
[laptop] extracting the audio track of interview.mp4 (no re-encode)
[laptop] transcript found: interview.words.json (the fake ASR replays it)
[1/7] decoding audio
[2/7] separating speech from background + transcribing (in parallel)
[3/7] finding who speaks when
      149 words -> 14 sentences -> 13 units
[4/7] reading the whole script: summary, glossary, cast sheet
[5/7] translating
[6/7] voicing and fitting 13 units
[7/7] mixing and writing subtitles
[laptop] muxing into the original video (video stream copied, not re-encoded)
```

Open `demo/interview.dub/interview.vi.mp4`. The default audio track is the "dub": a synthetic buzz per line, placed and stretched by the timing ladder over the faded original. The second audio track is the original. There are two subtitle tracks.

The demo script is a cooking-show interview between a host (S1) and an older guest (S2). It was written to exercise the hard cases:

| In the script | Exercises |
|---|---|
| "…, so the whole house smelled of smoke for two days, but I was far too proud…" (> 15 s) | DP splitting at commas |
| "Oh no!" (< 1 s) | tiny-unit merge |
| a 1.8 s pause | hard boundary |
| "Mr. Tu" | abbreviation is not a sentence end |
| "sir", "Mr." vs "Lan" | gender evidence from dialogue vs a bare name |

## 2. Run it again

```
calls:     37, cache hits: 37
```

Every step is memoised under a content hash of its inputs (see [architecture.md](architecture.md#the-call-context-and-the-cache)). Some things to try:

- Edit `configs/dub.yaml` `timing.rate_p95` and rerun. Only fitting, stretching, mixing and subtitles run again.
- Edit `src/dub/ops/subs.py` and rerun. Only `subs.render` runs again: CPU steps hash their own source code.
- Delete `dub-workspace/dub.sqlite` and rerun. Still 100% hits: SQLite is only an index.

## 3. Read the report

`dub-workspace/jobs/<job>/report.md` has:

- **Calls per op:** provider, calls, cache hits, run time.
- **Routing:**
  - which rule or source chose each provider;
  - every rejected candidate with its reason (e.g. `phowhisper_large_ct2: no manifest … (not installed in this version)`).
- **Timing:**
  - speed-up p50/p95/max;
  - which variants were used;
  - flags (`fast`, `overflow`);
  - the five units to review first.
- **Cast sheet:** speakers, relations with address terms, and open questions.
- **Subtitles:** cue count and cues that read too fast.

## 4. Look at the timeline

Drop `trace.json` on <https://ui.perfetto.dev>.

- Each lane (`cpu`, `llm`, `gpu · tts`, …) is a track.
- Overlapping calls get sub-tracks (`llm #2`), so you can see the parallelism: separation and ASR run together, and translation windows run concurrently.
- Cache hits show as instant markers.

## 5. Translate for real

```bash
uv run dub run demo/interview.mp4 --to vi --fake --use "mt.*=claude_cli"
```

This makes 3 Claude calls (about 40 s). From a real run of the demo:

```
## Cast sheet
- S1: name=Lan, gender=female (0.90), age=adult, role=host
- S2: name=Tư, gender=male (0.93), age=senior, role=guest
- S1 -> S2: [polite] {'vi': {'self_term': 'cháu', 'other_term': 'bác'}}
- S2 -> S1: [polite] {'vi': {'self_term': 'bác', 'other_term': 'cháu'}}
Needs confirmation:
- Should Mr. Tu be "bác" or "ông"? Default: bác/cháu.
```

and every line kept those forms of address, e.g. *"Bác cảm ơn cháu, Lan."*, *"Vậy món đầu tiên bác từng nấu là món gì vậy ạ?"*.

## 6. What is real and what is fake in M0

| Step | M0 | Real from |
|---|---|---|
| Audio extraction, decode, mux | real (ffmpeg) | — |
| Separation | fake: original at −20 dB | M1 (Mel-Band RoFormer) |
| ASR | fake: replays the transcript | M1 (Whisper / PhoWhisper) |
| Diarization | fake: speakers from the transcript | M4 |
| Segmentation | **real baseline** | M3 adds SaT |
| Context + translation | fake, or **real with `claude_cli`** | — |
| TTS | fake buzz | M1 (OmniVoice) |
| Timing fit, stretch, mix, subtitles | **real** (ffmpeg `atempo`) | M1 rubberband, M3 measured rates |

## 7. Tests

```bash
uv run pytest            # ~45 s locally; CI runs the same on Ubuntu with ffmpeg
```

| File | Covers |
|---|---|
| `test_keys_store.py` | Key stability and sensitivity, CAS dedupe/atomicity, memo corruption |
| `test_router.py` | Override, rules (first match), langpack chain, capability filter, draft incumbent |
| `test_segment.py` | Hard boundaries, abbreviations, DP split ≤ 15 s at commas, merges |
| `test_timing.py` | Syllable counting, variant choice, safe ratio, borrowing, stretch/flags, ripple locality |
| `test_subs.py` | Balanced lines, no orphan cues, CJK, no overlaps, SRT format |
| `test_claude_cli.py` | Prompt payloads, cast-sheet conversion, env scrubbing, error classification (fake CLI) |
| `test_pipeline.py` | End to end: valid MP4 with 2 audio + 2 subtitle tracks, 100% cache rerun, exact-duration engine, subtitles-only path |
| `test_architecture.py` | The core never imports providers; manifests resolve; every op has a fake |
