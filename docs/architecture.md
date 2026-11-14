# Architecture

## Goals

1. Low GPU cost: no idle GPU, no model reloads, batch where possible.
2. Maximum parallelism, low latency.
3. Easy to extend and to edit: adding a language or a model is a config change. Fixing one line re-renders one line.
4. Route per case: different models for different languages, content and line shapes.

The levers for speed and cost are scheduling, model residency, batching and model choice, not the programming language. 95–99% of the run time is inside CUDA/C++ kernels and ffmpeg, so everything here is Python.

## Where things run

```
LAPTOP                                   GPU POD (RunPod, from M1)            EXTERNAL
video file                               dubd: the pipeline (this repo)
  │ ffmpeg -vn -c:a copy (audio only)    workers: one process per model       Claude (claude -p,
  ├────────── audio (60–90 MB/h) ──────► family, all models resident in VRAM  your subscription)
  │                                      network volume: cache, weights
  ◄──── dubbed audio + subtitles ────────┤
ffmpeg -c:v copy: mux into the original video
```

- **The video never leaves the laptop.** Only the audio track is processed; the final mux copies the original video stream, so it takes seconds and never re-encodes.
- In M0 the pipeline runs on the laptop with fake providers. From M1 the same code runs on the pod with real models.

## The pipeline

`src/dub/core/pipeline.py`, phases from the design:

| Phase | Steps | Notes |
|---|---|---|
| A: analysis | decode → separate ∥ ASR → diarize → segment | whole file; GPU-bound |
| Barrier | `mt.context`: one LLM pass over the whole transcript | summary, glossary, **cast sheet** |
| B: per unit | translate windows → TTS → timing fit → stretch | units run concurrently |
| Tail | 30 s mix tiles → subtitles; on export: concat → loudness → AAC → (laptop) mux | tiles are cached one by one |

Granularity: sentence ⊇ dub unit ⊇ subtitle cue.

- A **dub unit** is one speaker and 1–15 s long. It is the unit we translate, voice, fit and edit.

## The call context and the cache

Every step goes through one doorway, `Ctx` (`src/dub/core/ctx.py`):

```python
words = await cx.call("asr.transcribe", TranscribeReq(...))                       # a model
seg   = await cx.local("text.segment", segment_ops.segment, SegmentReq(...), Segmentation)  # CPU
```

1. The router picks a provider chain for the op.
2. A **call key** is computed: `blake2b(canonical_json({op, provider, impl_version, weights_rev, params, seed, request}))`.
3. If `memo/<key>.json` exists and every file it references is in the CAS, the stored result is returned (a cache hit).
4. Otherwise the provider runs, its result is stored, and returned.

Consequences:

- **Resume** after a crash is just a rerun: finished steps are hits.
- **Early cutoff:** if a step produces byte-identical output, every key downstream is unchanged.
- **Cheap edits:** changing one unit's text changes only the keys that depend on it.
- **Code changes are tracked:**
  - CPU steps include a hash of their module source in the key.
  - Providers use `impl_version` from the manifest (`auto` = source hash, used for fakes).
  - LLM prompts are hashed into the key.

Storage (`src/dub/store/`):

| Store | Holds | Source of truth? |
|---|---|---|
| `cas/` | Files named by content hash (`<hash>.wav`), written atomically | yes |
| `memo/` | One JSON per call key: small results + CAS keys | yes |
| `dub.sqlite` | Index of jobs and calls | no, rebuildable |

## Routing

`src/dub/core/router.py`. For each call the router builds a chain of providers. First match wins:

1. CLI overrides: `--use mt.*=claude_cli`, `--fake`.
2. Rules in `dub.yaml` `routes:`, matched on facts about the job and the unit (`tgt_lang`, `unit.slot_ms`, `unit.words`, `unit.asr_conf`, …). A rule can `use` specific providers or `require` a capability.
3. The chain in the langpack: source language for separation, ASR and diarization; target language otherwise.
4. `defaults:` in `dub.yaml`.

Every candidate then passes the **capability filter**:

- serves the op;
- supports the language;
- `max_out_s` is long enough for the unit;
- not a `candidate` (experiments only);
- satisfies the rule's `require`;
- can actually run here (binary on PATH, module importable).

Each decision is written to `routes.jsonl` with the rejected candidates and why, and summarised in `report.md`. `dub doctor` prints the matrix for every language.

## Providers (plugins)

A provider is one adapter per model family, described by `configs/providers/<id>.yaml`:

```yaml
id: claude_cli
op: [mt.context, mt.translate, mt.rewrite]
impl: dub.providers.claude_cli:ClaudeCLI     # class(manifest).run(op, req, ctx) -> result
caps: {langs: "*", duration_control: none}
limits: {max_concurrency: 4}
on_error: {usage_limit: pause_then_retry, retry_wait_s: [60, 300]}
```

Rules, checked by `tests/test_architecture.py`:

- `run` is a pure function of (request, params, seed).
- The core (`core/`, `store/`, `contracts/`, `ops/`) never imports `dub.providers`. Adding a model never touches the core.
- Errors are classified (`usage_limit`, `auth`, `timeout`, `bad_output`, `crash`), so the core can pause and retry or fall back to the next provider in the chain.

Runtimes:

- `inproc` (M0): the class runs in a worker thread.
- `http` (M1): the same request body is sent to a worker process on the pod, which reads and writes the shared CAS.

## Data contracts

`src/dub/contracts/v1.py`. Every request and result is a pydantic model, exported with `dub schema`. Large data never travels inside a contract: audio is referenced by CAS key. The main contracts:

| Contract | Produced by | Used by |
|---|---|---|
| `MediaInfo` | decode | everything |
| `Stems` | separate | ASR, diarize, mix |
| `WordTimeline` | ASR | segment |
| `Diarization` | diarize | segment, context |
| `Segmentation` | segment | context, translate, dub |
| `ContextSheet` + `CastSheet` | `mt.context` | translate |
| `TranslationBatch` | `mt.translate` | dub |
| `SynthRes` | TTS | fit |
| `Placement` | timing fit | mix, subtitles |
| `SubtitleTrack` | subtitles | mux |

## Segmentation (priority 1)

`src/dub/ops/segment.py`.

1. **Hard boundaries:** a speaker change, or a pause of at least 1.5 s.
2. **Sentences:** split at terminal punctuation. Abbreviations ("Mr.", "Dr.", "e.g.") and initials are not sentence ends.
3. **Over-long sentences** (more than 15 s): split by dynamic programming. The cost favours cuts after commas and long pauses, and units close to 7 s.
4. **Tiny units** (under 1 s) are merged into a neighbour of the same speaker.

M3 replaces the punctuation and pause scores with SaT model probabilities and keeps the same DP.

## Timing fit (priority 2)

`src/dub/ops/timing.py`. The ladder puts quality first; changing speed is the last resort:

1. **Text first:**
   - The LLM returns `{short, target, long}` for each line.
   - A rate model (speech units per second, per language) predicts how long each variant takes to say.
   - Pick the variant within ±8% of the slot; otherwise the longest variant that fits; otherwise the shortest.
2. **Borrow silence:** start up to 0.2 s early or end up to 0.5 s late, under the **bounded ripple** rule:
   - The gap between two lines is split at its midpoint, minus a 120 ms minimum pause.
   - A line may only borrow the unused part of its neighbour's half.
   - That part is computed from the neighbour's natural length and slot, never from where the neighbour was placed. So editing line *i* only re-fits *i−1, i, i+1*.
3. **Exact-duration engines** (OmniVoice) receive `target_ms`, kept within `safe_ratio` [0.92, 1.08] of the natural length.
4. **Rewrite** (profile `final`): ask the LLM for a shorter line, at most 2 times.
5. **Stretch:**
   - pitch-preserving time stretch (ffmpeg `atempo` in M0; rubberband on the pod);
   - total speed-up target ≤ 1.10, hard cap 1.20;
   - beyond the cap the unit is flagged `overflow`.

Every placement records its speed-up `rate` and flags. The report lists the units to review first.

## Forms of address: the cast sheet

LLMs translate in windows of about 12 lines. Without shared knowledge of who is male or female, who is older, and how two people relate, each window guesses differently. Typical failures: the same two people switch from "anh/em" to "tôi/bạn"; a woman's line gets masculine agreement in Spanish; 她 becomes "anh ấy".

The barrier step `mt.context` reads the whole transcript once and returns a **cast sheet**:

- per speaker: name, gender + confidence, age band, role, style, evidence;
- per ordered pair: relation, formality, and target-language address terms (`self_term`, `other_term`), with `since_ms` because relations can change;
- the audience address term;
- people who are mentioned but do not speak;
- `needs_confirmation`: uncertain points, phrased as questions for the user.

Evidence ranks, strongest first:

1. user edits;
2. explicit dialogue (titles, kinship words, gendered words);
3. the voice model;
4. context.

Gender is never inferred from a name alone.

Each translation window receives only the cast rows of the speakers present (`CastSheet.rows_for`). Its cache key therefore changes only when those rows change. The rule-based address checker and the whole-script consistency pass arrive in M4.

## Human edits

Edits are stored on the laptop and sent with every run. Each kind enters the pipeline where it re-runs the least:
- split and merge are applied before translation, using windows cut on the unedited units;
- text replaces a translation;
- speaker, nudge and seed are applied after translation.

Together with per-tile mixing, a one-line fix redoes one TTS call and one or two tiles. See [m2-edit-loop.md](m2-edit-loop.md).

## Design rules

- The scheduler may change **when** work runs, never **what** is produced (ADR-014). Results are reproducible, so resume and golden tests work.
- No new abstraction without two concrete users.
- Defaults change only through measured comparisons (the "swap gate").
- Never infer gender from names.
- Everything in the repo is English; the code is Python only.
