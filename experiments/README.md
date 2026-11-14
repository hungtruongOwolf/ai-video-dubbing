# Experiments

Measurements that set defaults. A default changes only through a measured comparison (the
"swap gate" in the design), so every number in `configs/` that matters points to one of
these scripts and to its write-up in `docs/m3-segmentation-timing.md`.

| Script | Experiment | Runs on | Decides |
|---|---|---|---|
| `subs_layout.py` | Subtitle line breaking for Thai, Chinese, Japanese: M0 vs M3 | laptop (CPU) | the layout rules |
| `e4_segmentation.py text` | E4: rules vs SaT on 1997 NTREX sentences x 6 languages, with and without punctuation | pod, `tts` env (wtpsplit) | the SaT model and its threshold |
| `e4_segmentation.py speech` | E4 on real speech: OmniVoice reads NTREX, the langpack ASR transcribes, the pipeline segments | pod, `core` env + running workers | confirms E4; onset error of unit starts (ASR vs VAD snapping) |
| `tts_rate.py gen/mos/report` | E7 natural speaking rate, E3 safe squeeze range, E9 16 vs 32 steps | pod, `core` + `tts` envs + running workers | `units_per_s` per language, `safe_ratio`, draft `num_step` |

Data: [NTREX-128](https://github.com/MicrosoftTranslator/NTREX) (CC BY-SA 4.0), downloaded
on first use into `$DUB_EXP_DATA` (default `experiments/data/`, not committed). Results are
written to `$DUB_EXP_DATA/results/*.json`.

## Running on the pod

The workers must be up (`uv run dub up`). On the pod (`/workspace/dub`):

```bash
export DUB_EXP_DATA=/workspace/dub/experiments
cd /workspace/dub/repo
../envs/tts/bin/python  experiments/e4_segmentation.py text
../envs/core/bin/python experiments/e4_segmentation.py speech    # then: analyse, anywhere
../envs/core/bin/python experiments/tts_rate.py gen
../envs/tts/bin/python  experiments/tts_rate.py mos
../envs/core/bin/python experiments/tts_rate.py report
```

The scripts talk to the dubd workers over HTTP (`experiments/common.py: run_worker`), so the
models already loaded on the GPU are reused, and audio goes through the shared CAS.
