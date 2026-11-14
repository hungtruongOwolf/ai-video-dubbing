# M2: the edit loop

Goal: fix one line and hear the result in seconds, re-doing only that line, never the whole video.

## Using it

```bash
uv run dub review video.mp4 --to vi --from en --remote     # or without --remote (local, --fake)
```

The page opens at <http://127.0.0.1:8765>. Lines are sorted riskiest first. Click a line to:

- **▶ Original / ▶ Dub:** hear both, with 0.6 s of context on each side. The original is cut from your local video; the dub comes from the mixed tiles (on the pod with `--remote`).
- **Save text:** your line replaces the translation and is locked against automatic rewrites.
- **New take:** same text, next seed, a different reading.
- **Speaker:** reassign the line. A new speaker id gets its own cloned voice from its own lines. This is manual diarization until M4.
- **Slot ±0.1 s:** move the start or end of the time slot.
- **Split in two / Merge with next.**
- **Lock / Unlock;** every edit has an **undo** button.

Then press **Render changes**: only the touched lines and their 30 s tiles are redone. **Export video** also runs the final loudness pass and muxes the result next to your video.

The same edits from the command line:

```bash
uv run dub edit video.mp4 --to vi list
uv run dub edit video.mp4 --to vi text u0015410-S1 "Chú cảm ơn cháu Lan nhé."
uv run dub edit video.mp4 --to vi regen u0019950-S1
uv run dub edit video.mp4 --to vi speaker u0015410-S1 S2
uv run dub edit video.mp4 --to vi nudge u0021310-S1 -100 200
uv run dub edit video.mp4 --to vi split u0027750-S2 31200
uv run dub edit video.mp4 --to vi merge u0044720-S1
uv run dub edit video.mp4 --to vi undo
uv run dub run  video.mp4 --to vi --from en --remote      # renders and exports with the edits
```

## How it works

**Edits live on the laptop.**
- They are stored in `dub-workspace/edits/<audio>-<lang>.jsonl`: append-only, fsync'ed, and undo appends an `undo` record.
- Every run (local or remote) sends the full list with the job. So you can edit while the pod is off, and nothing is lost if the pod dies.
- Edits apply to every profile, so promoting a draft to final keeps your fixes.

**Each edit enters the pipeline where it re-runs the least** (`src/dub/ops/edits.py`):

| Edit | Applied | Re-runs |
|---|---|---|
| text | replaces the translation of that unit | its TTS, its tile, the target subtitles |
| seed (new take) | the TTS request | its TTS, its tile |
| speaker | after translation | the voice references, TTS of the lines whose voice changed |
| nudge | after translation | fit of the line and its neighbours, 1–2 tiles |
| split / merge | before translation | the translation window that holds the unit, the new units' TTS |
| lock | rewrite guard | nothing by itself |

Supporting rules:
- Translation windows are cut on the **unedited** units, and their context lines come from the unedited units too. A split in one window therefore never re-translates the others.
- Unit ids stay deterministic: `u…-S1` → `u…-S1.a` / `u…-S1.b`, and a merge gives `<a>+<b>`.

**Mixing in 30 s tiles** (`src/dub/ops/mix.py`).
- The timeline is cut on a fixed grid, independent of where clips land. Each tile mixes the bed slice plus every clip that overlaps it, and is cached on its own.
- Everything is sample-wise (fades, soft clipper), so tiles join without seams.
- Concatenation, EBU R128 loudness and AAC run only on export.

**Risk score** (`src/dub/ops/risk.py`).
- Signals, with hand-set weights: speed-up beyond 1.08, overflow, untranslated, rewrites, low ASR confidence, very short slots, translator notes.
- Edited lines drop down the list.
- M4 adds the QA gate (ASR back-check, speaker similarity) and learns the weights from your decisions.

**Remote round trips.** Each render costs two requests to the pod:
1. submit the job;
2. one long-poll that returns when the job ends, carrying all the small result files.

The pod reports whether it already has the audio, so nothing is uploaded twice.

## Measured (65 s two-voice clip, RTX 4090 in EU-RO-1, laptop in Vietnam)

| | Time |
|---|---|
| Pod-side job after a one-line edit (1 TTS + 1 tile) | 0.9–1.4 s |
| Edit → render done, as seen from the laptop | p50 2.3 s, max 2.6 s (after warm-up) |
| Edit → audible preview (render + preview download) | p50 3.5 s |
| Export (concat + loudness + AAC + download + mux) | 9.8 s |
| First boot of a fresh pod with everything cached on the volume | 75–107 s |

Every edit re-synthesised exactly one line, checked with the `redone` counter. The remaining ~1 s per render is two round trips through the Runpod proxy between Vietnam and Romania.

Re-export time for a 60-minute video has not been measured yet; it is dominated by the loudness pass and the ~85 MB AAC download.

## Tests

- `tests/test_edits.py`:
  - undo;
  - split/merge ids;
  - stale structure edits;
  - speaker/nudge/text/lock;
  - edit store durability;
  - tiles and the soft clipper;
  - risk ordering;
  - a text edit re-synthesising exactly one line;
  - undo returning to the cached take;
  - **kill-and-resume** (crash in the middle of TTS, rerun, byte-identical output);
  - the review page API.
- `tests/test_worker_remote.py`: the pod protocol, including uploads, the long-poll and the result bundle.
