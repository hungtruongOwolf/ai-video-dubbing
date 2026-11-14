#!/usr/bin/env bash
# Start command of the dubbing pod. Idempotent: everything heavy (uv, Python, ffmpeg, worker
# envs, model weights) is cached on the network volume, so only the first boot is slow.
set -uo pipefail
ROOT=${DUB_ROOT:-/workspace/dub}
mkdir -p "$ROOT"/bin "$ROOT"/logs "$ROOT"/envs
exec > >(tee -a "$ROOT/logs/bootstrap.log") 2>&1
echo "== bootstrap $(date -u +%FT%TZ) pod=${RUNPOD_POD_ID:-local} ref=${DUB_REF:-main}"

# The uv cache lives on the volume and venvs hold symlinks into it (see pod/setup_env.sh):
# one copy of torch for all worker envs. Never clean or prune this cache.
export UV_CACHE_DIR=${DUB_UV_CACHE:-$ROOT/uv-cache} UV_PYTHON_INSTALL_DIR=$ROOT/python UV_LINK_MODE=symlink
export HF_HOME=$ROOT/hf DUB_HF_CACHE=$ROOT/hf/hub DUB_MODELS=$ROOT/models
export PATH=$ROOT/bin:$HOME/.local/bin:$PATH
fail() { echo "!! bootstrap failed: $*"; sleep infinity; }  # keep the pod up so logs can be read

# 1. uv (Python package manager)
if [ ! -x "$ROOT/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/bin" UV_NO_MODIFY_PATH=1 sh \
    || fail "uv install"
fi

# 2. ffmpeg: static build (includes librubberband for high-quality time stretch)
if [ ! -x "$ROOT/bin/ffmpeg" ]; then
  tmp=$(mktemp -d)
  if curl -L --fail -s -o "$tmp/ff.tar.xz" \
      https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
     && tar -xJf "$tmp/ff.tar.xz" -C "$tmp"; then
    cp "$tmp"/ffmpeg-*-static/ffmpeg "$tmp"/ffmpeg-*-static/ffprobe "$ROOT/bin/"
  else
    echo "static ffmpeg download failed, falling back to apt"
    apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null || fail "ffmpeg"
  fi
  rm -rf "$tmp"
fi

# 3. code at the exact commit the laptop asked for
REPO_URL=${DUB_REPO:-https://github.com/hungtruongOwolf/ai-video-dubbing.git}
if [ -d "$ROOT/repo/.git" ]; then
  git -C "$ROOT/repo" fetch --quiet origin || fail "git fetch"
else
  git clone --quiet "$REPO_URL" "$ROOT/repo" || fail "git clone"
fi
git -C "$ROOT/repo" checkout --quiet --force "${DUB_REF:-origin/main}" || fail "git checkout ${DUB_REF:-}"

# 4. the core env runs dubd; worker envs are installed by dubd in the background
bash "$ROOT/repo/pod/setup_env.sh" core || fail "core env"

# 5. Claude Code CLI for the translation step, only when a subscription token is provided
if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && ! command -v claude >/dev/null; then
  curl -fsSL https://claude.ai/install.sh | bash || echo "claude CLI install failed (translation will fail)"
fi
unset ANTHROPIC_API_KEY  # never bill an API key instead of the subscription

export DUB_ROOT=$ROOT DUB_REPO_DIR=$ROOT/repo
# Hugging Face download locks live on the network volume. A download cut by a container
# restart leaves its lock held for good (the volume keeps the dead holder's lock), and the
# next worker waits on it forever. Nothing is running yet, so every lock is stale here.
rm -rf "$ROOT/hf/hub/.locks"
# M6 experiment (pod.local.yaml: mps: true): CUDA MPS lets the worker processes share the GPU
# concurrently instead of time-slicing it. Every CUDA process started below inherits it.
if [ "${DUB_MPS:-0}" = "1" ] && command -v nvidia-cuda-mps-control >/dev/null; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d && echo "== CUDA MPS daemon started" || echo "== CUDA MPS failed to start"
fi
echo "== starting dubd"
exec "$ROOT/envs/core/bin/python" -m dub.pod
