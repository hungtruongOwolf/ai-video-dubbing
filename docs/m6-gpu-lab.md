# M6: GPU lab

Optional experiments around the GPU. None of them blocks the product; each is a short note
and one line in the trend table. Unless stated otherwise: RTX 4090 (24 GB) in EU-RO-1, the
60.1-minute test speech of M5, en → vi.

## Trend table

| Date | Experiment | Setting | Result | Kept? |
|---|---|---|---|---|
| 2026-11-14 | Park models in CPU memory instead of unloading them (24 GB profile) | Whisper large-v3 (CTranslate2), Mel-Band RoFormer (torch) | Whisper: back on the GPU in 0.5 s instead of 9.7 s from the volume (3.6 GB freed); RoFormer: 0.1 s instead of 14.9 s (1.9 GB freed) | yes: `offload()`/`restore()` in the adapters, used by the worker's unload |
MPS_ROW
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

MPS_NOTE

## Community cloud

Network volumes attach to Secure Cloud pods only. A community pod would install the envs
and download the weights on every boot: from the measurements of this session, the envs take
~10.5 min to install from an empty cache and the weights (12.4 GB on the volume) a further
~1-2 min, so ~12.5 min before the first job, on a 40+ GB container disk. With the catalog
prices for an RTX 4090 (community $0.34/h, secure $0.89/h) a single 60-minute final would
cost about $0.15 there against $0.25 on a warm secure pod. But nothing survives the pod (no
cache for edits, no models), and the catalog showed no community RTX 4090 in stock when it
was read. Not built; worth revisiting for batch jobs of many videos.
