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

## M2: Edit loop (done)

- [x] Human edits: text, new take (seed), speaker, slot nudge, split, merge, lock, undo
- [x] Append-only, fsync'ed `edits.jsonl` on the laptop. Every run sends the full list, so edits work offline and survive pod loss.
- [x] Edits enter the pipeline where they re-run the least:
  - stable translation windows;
  - voice/timing edits after translation;
  - deterministic ids for split/merge units
- [x] 30 s mix tiles cached separately; concat + loudness + AAC only on export
- [x] Risk score per line (hand-set weights)
- [x] Review page (`dub review`): riskiest lines first, original vs dub playback, every edit, render, export
- [x] `dub edit` for the same edits from the terminal
- [x] Fewer round trips to the pod: no re-upload, a long-poll that carries the result bundle
- [x] Cache keys ignore fields at their default value, so adding an optional contract field keeps the cache
- [x] Kill-and-resume test: crash mid-TTS, rerun, byte-identical output

**Exit criteria:**

| Criterion | Result |
|---|---|
| Only the edited line is re-synthesised | yes. Every measured edit redid exactly one TTS call and one or two tiles. |
| Edit → render, p50 ≤ 2 s / p95 ≤ 6 s | p50 2.3 s, max 2.6 s after warm-up (pod 0.9–1.4 s + ~1 s of round trips Vietnam ↔ Romania). With the audible preview, p50 3.5 s. |
| Re-export ≤ 60 s for 60 min | not measured yet (65 s clip: 9.8 s). It will be measured with the M5 benchmark. |
| Kill and resume | yes: byte-identical output in the test suite |
| Offline edits | yes: edits live on the laptop |

**Changed from the original plan:**
- Edits are stored on the laptop instead of on the network volume. That makes the S3 export/import step unnecessary.
- The risk weights are hand-set until M4 brings QA signals.

## M3: Segmentation + timing (priorities 1 and 2) (done)

- [x] SaT boundary probabilities (`sat_12l_sm`) feed the sentence threshold and the DP (E4: text and speech arms, 6 languages)
- [x] Unit edges snapped onto the speech of the separated vocals (Whisper starts phrases 300–500 ms early)
- [x] Measured speaking rates for OmniVoice per language (E7), safe squeeze range (E3), 16 vs 32 steps (E9)
- [x] TTS clips trimmed to the voice; `.m4a` extraction for MP4 sources (fixes a 21 ms shift)
- [x] Dub-side alignment: every dubbed line is transcribed again; subtitle cues follow the new voice; CER in the risk score
- [x] Subtitle layout per script: Thai word breaking, CJK phrases + kinsoku, right-to-left marks, Netflix widths
- [x] `metrics.json` + "Sync and quality" report section; `dub bench seg --golden` for your own videos

**Exit criteria:**

| Criterion | Result |
|---|---|
| No unit longer than 15 s | yes (longest 9.9 s on the test clip; the DP guarantees it) |
| Onset error p95 ≤ 150 ms | test clip with known onsets: 10 ms (was 494 ms); E4 speech: en 150, es 30, vi 10, zh 20 ms; **ja/th ~0.5 s** on a few chunks |
| Speed-up p95 ≤ 1.15 | final 1.08; draft 1.15 |
| Thai line breaks: 0 errors | yes (M0 broke inside a word at 78% of its breaks) |
| Segmentation F1 ≥ 0.92 (talks) | **not met** on 11 real videos: en 0.85, es 0.83, ja 0.81, zh 0.76, vi 0.71 (read news on speech: en 0.92, es 0.93) |

**Changed from the original plan:**
- SaT runs in the `tts` env: the 50 GB volume could not hold a third copy of torch.
- Rates are per language (for OmniVoice), not per voice: a cloning TTS gives no natural
  target-language speech of the voice to learn from.
- E6 (ASR per source language) moved to M4. The LLM arm of E4 ran after M6: SaT stays (0.805 against 0.72-0.74 for the LLM on the 11 videos).
- Edit → render p50 went from 2.3 s to 3.3 s: the back-check and subtitle timing of the edited
  line (`backcheck: false` to skip).
- The volume quota counts hardlinks: the uv cache moved to the container disk; `dub gc` added.

Details: [m3-segmentation-timing.md](m3-segmentation-timing.md).

## M4: Speakers + routing + second engine (done)

- [x] Diarization: SpeechBrain ECAPA + clustering (`ecapa_cluster`, Apache-2.0, no HF token); E8 picked threshold 0.85 / min share 5%
- [x] Turns re-cut at pauses; turn-by-turn ASR with several speakers (speaker swaps 3 → 0 on the drama)
- [x] Unit purity check (risk score, `metrics.json`)
- [x] Reference voice bank: the most typical pure 3–10 s line per speaker
- [x] `--confirm-cast` mode; cast corrections (`dub edit … cast`, review page cast panel)
- [x] Rule-based forms-of-address checker per language (`address_generic` in the langpack)
- [x] Whole-script address consistency pass + re-translation with feedback (E10)
- [x] QA gate: ASR back-check, speaker similarity (relative to the speaker's own lines), truncation; deterministic k-best takes (`final`: 3)
- [x] Second TTS engine: VieNeu-TTS v3 Turbo (`tts2` env); E1 against OmniVoice
- [x] E6: ASR per source language, mix vs vocals
- [x] Voice age/gender model as cast-sheet evidence: added after M6 (`audio.voice_profile`, audeering wav2vec2, shown in `cast.md`)
- [ ] Blind A/B experiments: **not done** (E1 uses automatic measures: CER, speaker similarity, UTMOS)

**Exit criteria:**

| Criterion | Result |
|---|---|
| No speaker swaps | yes on the 4-person drama (42 lines) and the 2-person interview |
| Forms of address correct on ≥ 95% of lines in a 4-person video | 42 of 42 (100%) on the drama, en → vi, `final` |
| Switching the vi TTS takes ≤ 3 lines of config | yes: one line in `langpacks/vi.yaml` (chain) + `status: active` in the manifest |

**Changed from the original plan:**
- Diarization uses ECAPA + clustering instead of pyannote (its models are gated behind a
  Hugging Face token); pyannote and NVIDIA Sortformer were not compared.
- The "second engine with continuous batching": VieNeu batches the chunks of one line on the
  GPU; batching across lines is M5.
- Local reference clips (a reference per line, not per speaker) were not built: one typical
  clip per speaker passed QA on 97.8% of the drama's lines.
- Kotoba-Whisper (ja) cannot run in the pipeline (no word alignment in faster-whisper).
- The pod's envs are now symlinks into one uv cache (torch stored once): the volume had no
  room for a fourth copy of torch.

Details: [m4-speakers-routing.md](m4-speakers-routing.md).

## M5: Throughput (done)

- [x] Batching across units: continuous batching in the workers (OmniVoice, Whisper back-checks, ECAPA); OmniVoice 0.84 → 0.22 s per line
- [x] LLM in parallel with TTS: translated windows are voiced while the rest is translated; final rewrites happen before voicing on exact-duration engines
- [x] `dub bench speed` (stages, lanes, ops, GPU utilisation from dubd's `nvidia-smi` samples); `--fresh`; `--param` overrides
- [x] 24 GB GPU profile: phase-A models unloaded after phase A, cached GPU memory handed back, out-of-memory errors retried (18 failures → 0)
- [x] One source → many target languages: `dub run --to vi,es`; single flight shares the analysis between the jobs
- [x] Long-video fixes: ECAPA batches by length (out of memory at 60 min), master 114 → 43 s, subtitle layout 5× faster, units capped at 15 s after snapping

**Exit criteria** (60.1-minute speech, en → vi, RTX 4090):

| Criterion | Result |
|---|---|
| A 60-minute final finishes in ≤ 25 minutes | **14.1 min** (draft 7.8 min; draft before M5 13.4 min) |
| The GPU is busy ≥ 70% of billed time | **81% during the job**; 68% if a warm pod start is counted too. The idle timeout and pod boots are outside the job: several videos or languages per session amortise them. |

**Changed from the original plan:**
- "Two phases" became unloading phase-A models on GPUs under 40 GB, rather than separate
  phase-A and phase-B worker sets: everything still fits a 24 GB card that way.
- Batched OmniVoice is not bit-reproducible against the same line alone (see
  [m5-throughput.md](m5-throughput.md#reproducibility)); every take is cached, so reruns and
  edits are unaffected.

Details: [m5-throughput.md](m5-throughput.md).

## M6: GPU lab (optional, never blocks the product)

CUDA graphs vs continuous batching, MPS, pinned-RAM model swapping, local LLM, community-cloud profile. Each experiment = a short note + one line in the trend table ([m6-gpu-lab.md](m6-gpu-lab.md)).

- [x] Pinned-RAM model swapping: phase-A models parked in CPU memory (Whisper back in 0.5 s instead of 9.7 s, RoFormer 0.1 s instead of 14.9 s): **kept**
- [x] MPS: draft 473 → 442 s (−6.5%, one run each): **opt-in** (`mps: true`)
- [x] Community-cloud profile: priced from measurements, **not built** (no cache between pods; no stock when read)
- [ ] CUDA graphs: not tried (VieNeu already captures its frame steps; OmniVoice is diffusion over a whole line, few large kernels)
- [ ] Local LLM: not tried (translation quality is the Claude subscription's job; the GPU is already the bottleneck after phase A)

## M7: Quality pass (done)

The code of eleven open-source dubbing projects was read for ideas; each idea was measured
before it was kept ([m7-quality-pass.md](m7-quality-pass.md)). Reference hardware is now the
RTX PRO 4500 Blackwell (32 GB, $0.72/h): a 60-minute draft costs $0.108 there against $0.117
on the RTX 4090.

- [x] Separation at the model's 44.1 kHz with autocast: vocals SI-SDR +0.6 dB, 40% faster: **kept**
- [x] M5's RoFormer `batch_size` was ignored by audio-separator: **claim corrected, setting removed**
- [x] Silence inside OmniVoice's takes compensated per voice: LLM rewrites 140 → 25, speech 4.82 → 4.31 syl/s: **kept**
- [x] Per-line loudness from the original vocals: **kept**
- [x] Voice audition (candidate reference clips tried before voicing): whole-job voice similarity unchanged (0.599 → 0.600): **off by default**, available
- [x] Mix stage off the critical path (levels per line, export while subtitles render): 72 → 51 s: **kept**
- [x] Segmentation trials, decided on a larger set: Vietnamese punctuation transfer (PhoWhisper's words + a batched Whisper pass's punctuation) F1 0.687 → 0.779 on 8 videos (7 better), +27% ASR time: **kept**; zh Whisper prompts (−0.044 / −0.020 on 7 videos) and a minimum sentence length (±0): **rejected**
- [x] Worker restarts, versioned result keys, single-flight cancellation fix: **kept**
- [x] More evaluation videos: YODAS2 (CC BY YouTube videos with uploader subtitles), +6 vi and +5 zh after a language and transcript check (`experiments/yodas_videos.py`); Chinese scored in simplified characters
- [ ] Subtitle break penalties, rotating reference clips across retakes, rewrite candidates: small expected gains, not done

## M8: One pass (done)

A video goes in and the final dub comes out with no manual correction. Test case: a 5-minute
zh short drama from TikTok (burned-in subtitles, ten speaking parts, a song) into Vietnamese,
whose first dub needed three or four rounds of edits ([m8-one-pass.md](m8-one-pass.md)).

- [x] Who says each line, read from the script by the LLM and checked against the voices
  (`text.speakers`): speaker changes that change voice 4% (default clustering) / 22% (finer
  re-clustering) → 90%, precision 87%; the 4-person drama keeps 0 swaps: **kept**
- [x] Burned-in subtitles read on the laptop (OCR) correct the transcript (61 characters on
  the drama; ASR vs subtitles CER 8.8%); a cue's line spans its words aligned by characters: **kept**
- [x] Songs and background voices left original (the LLM's character kinds); uncaptioned
  answers are dubbed: **kept**
- [x] Voice references: quiet edges, pooled short lines, heard again by the ASR, cleanest of 3
  candidates; OmniVoice fade 100 → 10 ms: **kept**
- [x] Back-check text a short clip cannot hold is dropped (PhoWhisper invents sentences on
  sub-second clips): **kept**
- [x] Short lines planned 1.6x longer for OmniVoice (4-6 syllables understood 67% → 96%): **kept**
- [x] Vietnamese lines of ≤ 6 syllables voiced by VieNeu (understood 35% → 64%, 3 syllables 17% → 83%): **kept**
- [x] Accent retry: a line whose every take sounds like the source language gets a VieNeu take
  (heard as Chinese: OmniVoice 22%, VieNeu 3.7%; guidance 3.0 made it worse): **kept**
- [x] Residual risks first in the report, one summary line at the end of a run: **kept**
- [x] Output video: the dub is the only audio track: **kept**
- [x] One-pass run of the drama: QA 0.84 → 0.90, CER p95 0.30 → 0.14, speed-up p95 1.02,
  14 of 86 lines worth a listen, 0 left failing for an accent
- [ ] Not done: an LLM pass correcting the transcript without subtitles, VoiceStudio's
  measured-duration rewrite, a listening test

## M9: Faster runs, better lines (done)

Write-up: [m9-faster-better.md](m9-faster-better.md).

- [x] VieNeu loaded with the pod (first calls of a job waited 130-196 s): cartoon job 175 s -> 71 s, drama voicing 227 s -> 95 s
- [x] Jobs start while workers load; workers serve while they load (first-use order); envs started at once; the pod rented while the video is read
- [x] Accent check by language detection only (40/40 same answers, 89 vs 460 ms a take)
- [x] Laughs, cries and fillers keep the original audio (langpack `interjections`)
- [x] Speakers: 3 readings voted, on Fable (B-cubed F1 0.711 -> 0.782, speaker changes recall 0.82 -> 0.88); disagreements flagged
- [x] Translation model: Sonnet kept (blind judges: even with Opus and Fable)
- [x] Wordings voiced and measured for VieNeu; QA retry on OmniVoice; worker lock race fixed
- [x] Drama: QA passed 0.899 -> 0.989, lines worth a listen 14/86 -> 11/96
- [ ] Not done: weights on faster storage (cold start ~4 min), who speaks from the picture, listening test

## M10: Expert review (done)

Write-up: [m10-expert-review.md](m10-expert-review.md). Seven read-only reviewers (engineering
manager, researcher, GPU, CUDA, speech quality, dubbing translator, audio post-production).

- [x] Voicing 87.8 s -> 74.5 s on the same translations (QA at once, cheap calls first, 256 threads,
  VieNeu batches, back-check and ECAPA in their own workers); parallel retakes and 48 in flight
  measured and rejected
- [x] Several videos on one pod (`dub run A B`): two videos 323 s vs 379 s one after another
- [x] Laughs, cries, screams between lines kept (no words heard in them); stems rebalanced
- [x] Prompts v2 (blind judges prefer v2 ~55%), variants that neither pad nor clip, VieNeu only in
  short slots (lines outside 0.7-1.2 fill: 28 -> 15 of 90)
- [x] Address, risk-list, voice-gate and flaky-test fixes; GPU power and batch histograms per job
- [ ] Not done: OmniVoice FlashInfer, Qwen3-ASR for zh, emotion-matched references, a back-check
  that knows the cast's names, envs on container disk, listening page

## M11: Phrase timing (done)

Write-up: [m11-phrase-timing.md](m11-phrase-timing.md).

- [x] A take cut at its own silences and re-timed so it pauses where its actor pauses (signal
  only, no new translation or take): drama windows with the dub silent while the actor speaks
  112 -> 63, cartoon windows under the original 15 -> 10
- [x] Re-timed takes heard again; the take as it was stays if the back-check misses it
- [x] Before it, M10 follow-up: kept originals cover their real sound (a song squeezed into
  300 ms), underfilled lines try fuller wordings judged by sync error in ms, final particles
  read as the same by QA, a misheard rewrite never replaces a good take, `dub run` updates a
  pod on older code
- [ ] Not done: cut points between words, phrase-aware translation

## M12: Voices by delivery (done)

Write-up: [m12-emotion-refs.md](m12-emotion-refs.md).

- [x] A pool of clean clips per speaker; each line voiced from the clip closest to its own
  delivery (pitch, range, level, voicing): drama pitch correlation with the source -0.10 ->
  0.46, pitch spread 0.26 -> 0.81 of the source's, QA passed 0.966 -> 0.978
- [x] Prosody agreement measured in every job (`prosody_*` in metrics.json)
- [x] Worker processes load one model at a time (a thread race broke OmniVoice)
- [ ] Not done: a learned emotion embedding, pitch range and level that follow the source

## M13: Emotion from emotion2vec (done)

Write-up: [m13-emotion2vec.md](m13-emotion2vec.md).

- [x] `audio.emotion`: emotion2vec+ large through FunASR, in its own env and worker
- [x] Each line's clip picked by prosody, emotion or both (`voice.style_by`, default both):
  on the drama emotion agreement 0.17 -> 0.24 at the same pitch agreement (0.45) and no new
  QA failure; emotion alone failed 3 more lines
- [x] Emotion agreement measured in every job (`emotion_cos`, `emotion_label_match`)
- [ ] Not done: a blind listening test of the modes, emotion in the QA gate
