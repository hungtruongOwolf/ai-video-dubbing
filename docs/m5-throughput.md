# M5: throughput

How long a real hour of video takes, where the time goes, and what made it shorter. All
numbers are from the pod (RTX 4090, 24 GB, 48 vCPUs) dubbing the same video, measured with
`dub bench speed --fresh` (every step recomputed).

**Test video:** the 2013 State of the Union address (Wikimedia Commons, public domain),
60.1 minutes, English, one main speaker with long applause breaks: 6907 words, 404
sentences, 418 dub units. Target: Vietnamese.

## Measuring

- `dub bench speed VIDEO --to vi --remote [--profile final] [--fresh] [--json F]` dubs the
  video and splits the wall time into stages (the `[k/7]` marks on the trace), lanes ("gpu ·
  tts", "llm", "cpu"...) and ops; `dub bench speed JOB_DIR` re-reads a finished job.
- dubd samples `nvidia-smi` every 0.5 s during each job (`gpu.csv` in the job folder); the
  trace records its wall clock, so each stage gets its GPU utilisation.
- `--fresh` ignores everything stored before the job; what the job computes itself is
  reused within the job (voicing ahead depends on it).

RESULTS_PLACEHOLDER
