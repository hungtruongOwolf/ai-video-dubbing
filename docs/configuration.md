# Configuration

Everything that decides *what* runs lives in `configs/`. The code decides *how*.

```
configs/
  dub.yaml                 profiles, timing, mix, translation windows, defaults, routing rules
  langpacks/<lang>.yaml    one file per language
  providers/<id>.yaml      one manifest per model / adapter
  prompts/*.md             LLM prompts (hashed into the cache key)
```

Set `DUB_CONFIG_DIR` or pass `--config-dir` to use another folder.

## Add a language

Create `configs/langpacks/<lang>.yaml`:

```yaml
lang: ko
name: Korean
iso639_2: kor
speech_unit: char        # syllable | char | word: what the rate model counts
units_per_s: 7.0         # natural TTS speaking rate (see experiments/tts_rate.py, E7)
joiner: " "              # "" for languages written without spaces
segment: {max_unit_s: 15, target_unit_s: 7, min_unit_s: 1.0, hard_pause_s: 1.5, threshold: 0.5}
subs: {cpl: 18, cps: 12, max_lines: 2}  # cpl in display width (CJK half-width counts half)
address_generic: []     # pronouns that are wrong where a pair uses kinship terms (vi: tôi, bạn...)
chains:
  asr.transcribe: [whisper_large_v3]
  tts.synthesize: [omnivoice]
```

Languages without a langpack use `langpacks/default.yaml`.

- `address_generic` feeds the forms-of-address checker: a line with one of these words is
  flagged when the cast sheet gives that pair kinship or title terms. Leave it empty for
  languages without such a system.

- `segment.threshold` applies to *calibrated* boundary probabilities: every boundary provider
  maps its own decision threshold to 0.5 (SaT: `threshold` in its manifest, chosen by E4).
  Lower it for a language where sentences are under-split.
- Subtitle layout needs no setting: line breaks follow the script (spaces; BudouX phrases for
  zh/ja; dictionary words for Thai), see `src/dub/ops/script.py`.

## Add a model

1. Write an adapter class in `src/dub/providers/` with `__init__(self, manifest)` and `run(self, op, req, ctx)`. The request and result types for each op are in `src/dub/contracts/v1.py`. `src/dub/providers/fake.py` is the simplest example.
2. Add `configs/providers/<id>.yaml`:

```yaml
id: my_tts
op: [tts.synthesize]
impl: dub.providers.my_tts:MyTTS
impl_version: "1"          # bump when a code change alters the output ("auto" = source hash)
weights: {hf: org/model, revision: "<commit sha>"}
params: {steps: 32}        # part of the cache key
caps:
  langs: [en, zh, vi]      # or "*"
  duration_control: exact  # exact | relative | rate | none | preserving
  max_out_s: 15
  needs_ref_text: false
limits: {max_concurrency: 4}
status: candidate          # candidates are used only in experiments until they win the swap gate
```

3. Reference it from a langpack chain, a rule, or `--use tts.synthesize=my_tts`.

`dub doctor` shows whether it can run here, and why not.

## Add a routing rule

Rules in `dub.yaml` are checked before the langpack chain; the first match wins.

```yaml
routes:
  tts.synthesize:
    - {id: short-line, when: {unit.slot_ms: {lt: 1300}}, require: {duration_control: exact}}
    - {id: zh-drama,   when: {tgt_lang: zh, profile: final}, use: [fish_s2_pro, omnivoice]}
```

`when` facts available today:

- job: `src_lang`, `tgt_lang`, `profile`;
- unit (for `tts.synthesize`): `unit.slot_ms`, `unit.words`, `unit.asr_conf`.

Conditions:

- a value means equality;
- a list means membership;
- a dict uses operators: `{lt, lte, gt, gte, eq, ne, in}`.

`use` replaces the candidates; `require` filters them by capability.

## Profiles

```yaml
profiles:
  draft: {tts_chain: incumbent_only, llm_rewrite: false, max_rewrites: 0}
  final: {tts_chain: langpack,       llm_rewrite: true,  max_rewrites: 2, takes: 3}
```

A profile only changes what happens after the text layer. Lines you fix in a draft survive the switch to final.

`params` lays provider parameters over the manifest's, per profile. Drafts use it for
OmniVoice's diffusion steps (experiment E9):

```yaml
  draft: {tts_chain: incumbent_only, llm_rewrite: false, max_rewrites: 0, params: {omnivoice: {num_step: 16}}}
```

`backcheck: true` (the default in both) transcribes every dubbed line again with the target
language's ASR: subtitle cues then follow the new voice, and the line's CER feeds the risk
score. Set it to `false` to save one ASR call per line.

`takes` (final: 3) is how many seeds a line may try until a take passes the QA gate; the
best-scoring take is kept when none passes. With `takes: 1` the gate still runs and its
verdict goes to the report and the review page. The final profile also re-translates
windows flagged by the forms-of-address checker (up to two rounds).

## QA gate

```yaml
qa:
  cer_max: 0.15        # back-check character error rate
  sim_ratio_min: 0.5   # voice similarity, relative to the speaker's own lines
  other_margin: 0.05   # fail when the take sounds more like another speaker
  voice_min_ms: 1200   # shorter takes are not judged on voice
  tail_min: 0.6        # share of the line's last quarter the ASR heard (truncation)
```

Why these values: [m4-speakers-routing.md](m4-speakers-routing.md#voices-and-the-qa-gate).

## Cast corrections

The cast sheet the LLM writes can be corrected; corrected rows are locked and only the
translation windows of those speakers are translated again:

```bash
dub run VIDEO --to vi --confirm-cast                 # stop after the cast sheet, print it
dub edit VIDEO --to vi cast S2 gender=female name=Linda
dub edit VIDEO --to vi cast S1>S2 self_term=mẹ other_term=con
dub run VIDEO --to vi                                 # the analysis is reused
```

The review page has the same fields in its cast panel.

## Switch the TTS engine of a language

Engines are chosen per language by the langpack chain (profile `final`) or by `defaults`.
To voice Vietnamese with VieNeu-TTS instead of OmniVoice, in `configs/langpacks/vi.yaml`:

```yaml
chains:
  tts.synthesize: [vieneu_v3_turbo, omnivoice]   # first runnable wins, OmniVoice as fallback
```

and in `configs/providers/vieneu_v3_turbo.yaml` set `status: active` (a `candidate` is
only used when named explicitly: `--use tts.synthesize=vieneu_v3_turbo`). Its env (`tts2`) is
installed by dubd on the next `dub up`.

## Throughput

```yaml
throughput:
  units_in_flight: 24          # lines voiced at the same time; bounds the TTS batch
  free_phase_a_models: auto    # unload separation / source ASR / SaT after phase A (GPUs < 40 GB)
```

A provider batches when its manifest allows it and its adapter has `run_batch`:

```yaml
limits: {max_concurrency: 32, max_batch: 16}   # calls in flight per job / largest batch
```

One run can override provider parameters without editing a manifest (the values are part of
the cache keys, so the variant is cached apart):

```bash
dub run VIDEO --to vi --remote --param whisper_large_v3.batch_size=16
dub bench speed VIDEO --to vi --remote --profile final --fresh   # time it, nothing from cache
dub run VIDEO --to vi,es,ja --remote                             # three languages, one analysis
```

## Timing

See `timing:` in `dub.yaml` and the ladder in [architecture.md](architecture.md#timing-fit-priority-2).
- `strict: true` halves the borrowing limits, for content that must stay tightly in sync.

## Prompts

`configs/prompts/mt_context.v1.md`, `mt_translate.v1.md` and `mt_rewrite.v1.md` are the system prompts of the `claude_cli` provider.

- Editing a prompt changes the cache key, so the affected calls re-run.
- Keep the old file and create `v2` if you want to compare the two versions.
