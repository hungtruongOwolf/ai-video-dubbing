# M1: the GPU pod

M1 moves the models to a rented GPU. The laptop still owns the video and does the final mux; the pod receives only the audio track.

```
laptop                                         Runpod pod (GPU, network volume at /workspace)
------                                         ------------------------------------------------
dub run video.mp4 --to vi --remote
  ffmpeg -vn -c:a copy (audio only) ─── HTTPS proxy ───► dubd (pipeline, port 8000)
                                                           ├─ worker "sep"  Mel-Band RoFormer
                                                           ├─ worker "asr"  Whisper large-v3, PhoWhisper
                                                           ├─ worker "tts"  OmniVoice
                                                           └─ claude -p     context + translation
  ◄──────────── dubbed audio (AAC) + subtitles + report ──┘
  ffmpeg -c:v copy (mux into the original video)
```

## One-time setup

1. **Runpod account with credit.** A network volume (50 GB ≈ $3.50/month) plus GPU hours.
2. **API key** for the laptop. On the [Credentials page](https://console.runpod.io/user/credentials), create a key with *All* permissions, then store it as the user environment variable `RUNPOD_API_KEY`:

   ```powershell
   $s = Read-Host "Runpod API key" -AsSecureString
   [Environment]::SetEnvironmentVariable("RUNPOD_API_KEY", [Net.NetworkCredential]::new("", $s).Password, "User")
   ```

3. **Network volume** in a data center that has the GPUs you want, for example with the Runpod console, `runpodctl`, or the Runpod MCP tools. Put its id and data center in `pod.local.yaml` at the repository root (git-ignored):

   ```yaml
   volume_id: <id>
   data_center: EU-RO-1
   gpus: [NVIDIA GeForce RTX 4090, NVIDIA RTX PRO 4500 Blackwell, NVIDIA L4]
   ```

4. **Claude subscription token** for translation on the pod (optional; without it, translate with `--use "mt.*=fake_mt"`):
   1. run `claude setup-token` in a terminal;
   2. create a Runpod secret named `claude_code_oauth_token` with the printed token.

   `dub up` passes it to the pod as `CLAUDE_CODE_OAUTH_TOKEN`; the value never touches the repository.

## Daily use

```bash
uv run dub up                                   # rent the pod, wait until every model is loaded
uv run dub run video.mp4 --to vi --from en --remote
uv run dub status                               # price, uptime, worker status
uv run dub down                                 # terminate it (or let it idle out after 10 min)
```

- `dub run --remote` starts a pod by itself when none is running.
- `--from` matters for the source language: `--from vi` routes ASR to PhoWhisper, other languages and `auto` use Whisper large-v3.

## What happens on the pod

`dub up` creates a pod from a stock image (`runpod/pytorch`, used only for its CUDA userland). The pod gets:
- the network volume mounted at `/workspace`;
- port 8000 exposed through the Runpod HTTPS proxy;
- a start command that downloads `pod/bootstrap.sh` at the exact git commit of your laptop. The commit must be pushed first; `dub up` checks this.

`pod/bootstrap.sh` (idempotent, everything heavy is cached on the volume):

1. installs `uv` and a static `ffmpeg` (with librubberband) into `/workspace/dub/bin`;
2. clones or updates the repository at that commit into `/workspace/dub/repo`;
3. creates the `core` env and installs the Claude Code CLI when a token is present;
4. starts `dubd`.

`dubd` (`src/dub/pod/server.py`) then:

- installs each worker env from `pod/envs/<env>.txt` into `/workspace/dub/envs/<env>`. An env whose requirements are unchanged is skipped.
- starts one worker per env. Each worker preloads its models and serves them on `127.0.0.1:710x`.
- accepts uploads in parts (the proxy limits requests to 100 MB), runs jobs, and serves the results.
- terminates the pod after `idle_minutes` without work, using the pod-scoped key Runpod injects. It never does this while a job runs or while the first boot is still installing.

Volume layout:

```
/workspace/dub/
  bin/        uv, ffmpeg, ffprobe
  python/     Python builds managed by uv
  uv-cache/   wheel cache shared by all envs
  envs/       core, asr, sep, tts
  repo/       the code at the requested commit
  hf/         Hugging Face cache (Whisper, PhoWhisper, OmniVoice weights)
  models/     audio-separator models
  ws/         CAS + memo + job folders (the pipeline cache)
  logs/       bootstrap.log, <env>.install.log, <env>.log
```

Because the pipeline cache lives on the volume, a rerun on any later pod is served from the cache.

## Models in M1

| Op | Provider | Env | Notes |
|---|---|---|---|
| `audio.separate` | `mel_roformer` | sep | Kim's vocals Mel-Band RoFormer; output resampled to 48 kHz |
| `asr.transcribe` | `whisper_large_v3`, `phowhisper_large_ct2` | asr | faster-whisper, word timestamps, VAD filter. PhoWhisper loads on first use. |
| `audio.diarize` | `single_speaker` | (inproc) | one voice for the whole file until M4 |
| `voice.extract_refs` | CPU op | — | one 3–10 s reference clip per speaker from the vocals stem |
| `mt.*` | `claude_cli` | (inproc) | needs the `claude_code_oauth_token` secret |
| `tts.synthesize` | `omnivoice` | tts | zero-shot clone from the reference clip, exact `duration` |
| stretch | CPU op | — | ffmpeg `rubberband` filter (pitch-preserving), `atempo` fallback |
| encode | CPU op | — | AAC 192 kb/s `.m4a` for a small download |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no GPU available right now` | The data center has no free card of any type in `gpus`. The catalog's "LOW" stock is not a guarantee. Retry later, add GPU types, or move the volume to a data center with more stock. |
| `dubd: not reachable yet` for a long time | Look at the pod logs (`bootstrap.log`). A failed bootstrap keeps the container alive so the logs can be read. |
| a worker shows `failed` | `/workspace/dub/logs/<env>.log` has the traceback. Fix `pod/envs/<env>.txt`, push, and start a new pod. |
| Python requests to the proxy fail but curl works | The Runpod proxy rejects Python's default User-Agent; the client sends `dub-cli/0.1`. |
| translation fails on the pod | Check that the secret `claude_code_oauth_token` exists, and that `dub up` did not warn about it. |
