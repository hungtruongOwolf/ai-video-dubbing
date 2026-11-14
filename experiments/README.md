# Experiments

Measurements that set defaults. A default changes only through a measured comparison (the
"swap gate" in the design), so every number in `configs/` that matters points to one of
these scripts and to its write-up in `docs/m3-segmentation-timing.md` or
`docs/m4-speakers-routing.md`.

| Script | Experiment | Runs on | Decides |
|---|---|---|---|
| `subs_layout.py` | Subtitle line breaking for Thai, Chinese, Japanese: M0 vs M3 | laptop (CPU) | the layout rules |
| `e4_segmentation.py text` | E4: rules vs SaT on 1997 NTREX sentences x 6 languages, with and without punctuation | pod, `tts` env (wtpsplit) | the SaT model and its threshold |
| `e4_segmentation.py speech` | E4 on real speech: OmniVoice reads NTREX, the langpack ASR transcribes, the pipeline segments | pod, `core` env + running workers | confirms E4; onset error of unit starts (ASR vs VAD snapping) |
| `real_videos.py fetch/run [--param P.K=V --tag T]` | Segmentation F1 on 11 real videos (Wikimedia Commons, human subtitles; silver labels in `golden/` for zh/ja); `--param`/`--tag` score a variant (e.g. batched Whisper) | laptop + pod (`dub bench seg --remote`) | where segmentation still fails; whether a faster ASR mode costs accuracy |
| `tts_rate.py gen/mos/report` | E7 natural speaking rate, E3 safe squeeze range, E9 16 vs 32 steps | pod, `core` + `tts` envs + running workers | `units_per_s` per language, `safe_ratio`, draft `num_step` |
| `drama.py make/fetch` | A 4-person family scene (`drama.json`: 42 lines, who speaks to whom, the Vietnamese address table), voiced with 4 cloned voices, with the truth of who speaks when | pod + laptop | test material of E8 and E10 |
| `e8_diarization.py` | E8: who speaks when; ECAPA clustering threshold sweep vs `single_speaker` on the drama, the interview clip and the AMI meeting ES2004a | pod, `core` env + running workers | `ecapa_cluster` threshold and min cluster share |
| `e10_address.py JOB` | E10: speaker swaps and forms of address of a dubbed drama run (en -> vi) | laptop | the address checker, the translation prompt rules |
| `e6_asr.py prep/run/analyse` | E6: ASR model x mix/vocals on the 11 real videos; Vietnamese numbers and largest differences | laptop + pod | the ASR chain per language, mix vs vocals |
| `e1_tts.py gen/mos/report` | E1 (+E2): OmniVoice vs VieNeu-TTS for Vietnamese, cloning 5 non-Vietnamese voices: back-check CER, speaker similarity, UTMOS, speed | pod, `core` + `tts` envs + running workers | the vi TTS default |
| `m6_offload.py` | M6: load vs park-in-CPU-memory times of Whisper and RoFormer | pod, worker envs | parking phase-A models between jobs |
| `timing_replay.py JOB_DIR` | Replays a finished job's timing fit with other borrowing limits (no GPU, no TTS) | laptop | `late_end_ms`, `early_start_ms` |
| `tts_silence.py [--ref KEY:TEXT]` | M7: silence OmniVoice leaves inside the requested duration, per reference voice, as is / tight reference / tight + pad | pod, `core` env + running workers | the per-voice silence pad |
| `sep_quality.py run --long-key KEY` | M7: separation SI-SDR on 6 synthetic mixes (clean speech over public-domain music) and speed on the 60-min speech, per sample rate and precision | pod, `core` env + running workers | `mel_roformer` sample rate and precision |

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
../envs/core/bin/python experiments/drama.py make
../envs/core/bin/python experiments/e8_diarization.py --interview-key KEY
../envs/core/bin/python experiments/e6_asr.py run --input KEY        # KEY from `e6_asr.py prep`
../envs/core/bin/python experiments/e1_tts.py gen
TORCH_HOME=/tmp/torch ../envs/tts/bin/python experiments/e1_tts.py mos
../envs/core/bin/python experiments/e1_tts.py report
```

`dub gc --remote` deletes experiment inputs uploaded to the pod's cache (nothing references
them): run `prep` again after a GC.

The scripts talk to the dubd workers over HTTP (`experiments/common.py: run_worker`), so the
models already loaded on the GPU are reused, and audio goes through the shared CAS.
