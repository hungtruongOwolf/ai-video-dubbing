# M6: GPU lab

Optional experiments around the GPU. None of them blocks the product; each is a short note
and one line in the trend table. Unless stated otherwise: RTX 4090 (24 GB) in EU-RO-1, the
60.1-minute test speech of M5, en → vi.

## Trend table

| Date | Experiment | Setting | Result | Kept? |
|---|---|---|---|---|
| 2026-11-14 | Park models in CPU memory instead of unloading them (24 GB profile) | Whisper large-v3 (CTranslate2), Mel-Band RoFormer (torch) | Whisper: back on the GPU in 0.5 s instead of 9.7 s from the volume (3.6 GB freed); RoFormer: 0.1 s instead of 14.9 s (1.9 GB freed) | yes: `offload()`/`restore()` in the adapters, used by the worker's unload |
| 2026-11-14 | CUDA MPS for the GPU workers (4 processes on one GPU) | draft, 60.1 min, fresh, one run each | 442 s with MPS against 473 s without (−6.5%): separate ∥ transcribe 165 vs 179 s, voicing 116 vs 134 s | opt-in only (`mps: true` in pod.local.yaml): one run per setting |
| 2026-11-14 | Community-cloud profile (no network volume) | RTX 4090: community $0.34/h, secure $0.89/h (catalog) | estimated per 60-min final: secure, warm pod ≈ 17 min ≈ $0.25 (+ volume $3.50/month); community, cold pod ≈ 12.5 min of installs and downloads + 14 min ≈ $0.15, but no cache between pods and no community 4090 in stock when read | no: not built |

## Models parked in CPU memory

The 24 GB profile of M5 frees the GPU memory of the models only phase A uses. Unloading
them made the next video pay their full load from the network volume again. Measured on
an idle GPU (`experiments/m6_offload.py`):

| Model | Load from the volume | To CPU memory | Back to the GPU | GPU memory freed |
|---|---|---|---|---|
| Whisper large-v3 (CTranslate2, fp16) | 9.7 s | 2.1 s | 0.5 s | 3.6 GB |
| Mel-Band RoFormer (torch) | 14.9 s | 0.6 s | 0.1 s | 1.9 GB |

Adapters that can do it now have `offload()` and `restore()` (Whisper/PhoWhisper,
RoFormer, SaT); the worker's unload parks those and restores them on their next call. The pod
has far more CPU memory than these models need.

## CUDA MPS

Four worker processes share the GPU (asr, sep, tts, tts2); without MPS the driver time-slices
between their CUDA contexts. With the MPS daemon started before dubd (`pod/bootstrap.sh`,
`DUB_MPS=1`), their kernels can run side by side:

| 60.1 min draft (fresh) | Without MPS | With MPS |
|---|---|---|
| total | 473 s | **442 s** |
| separate ∥ transcribe | 179 s | 165 s |
| voice and fit | 134 s | 116 s |
| GPU utilisation | 73% | 71% |

A 6.5% gain where two models compete (stages 2 and 6), from one run per setting: the second
run without MPS failed when the volume filled up (see the M5 notes), so the run-to-run spread
is unknown. MPS also ties the workers' fate together (a crashed client can take the daemon's
other clients with it). It stays an opt-in switch until more runs confirm the gain.

Finding the switch did not work at first exposed a bug: a pod's start command downloads
`bootstrap.sh` at the commit the pod was created with, and `dub up` only changed `DUB_REF`, so
no later change to `bootstrap.sh` had ever reached a running pod (the stale Hugging Face lock
cleanup of M4 included). `dub up` now updates the start command too, and the bootstrap
continues with the copy of itself in the commit it checks out.

## Community cloud

Network volumes attach to Secure Cloud pods only. A community pod would install the envs
and download the weights on every boot: from the measurements of this session, the envs take
~10.5 min to install from an empty cache and the weights (12.4 GB on the volume) a further
~1-2 min, so ~12.5 min before the first job, on a 40+ GB container disk. With the catalog
prices for an RTX 4090 (community $0.34/h, secure $0.89/h) a single 60-minute final would
cost about $0.15 there against $0.25 on a warm secure pod. But nothing survives the pod (no
cache for edits, no models), and the catalog showed no community RTX 4090 in stock when it
was read. Not built; worth revisiting for batch jobs of many videos.
