#!/usr/bin/env bash
# Usage: setup_env.sh <env>
# Creates or updates the uv venv /workspace/dub/envs/<env> from pod/envs/<env>.txt.
# The heavy install is skipped when the requirements have not changed since last time;
# our own package is always re-linked (editable install, no dependency resolution).
set -euo pipefail
ROOT=${DUB_ROOT:-/workspace/dub}
NAME=$1
REPO=$ROOT/repo
REQ=$REPO/pod/envs/$NAME.txt
VENV=$ROOT/envs/$NAME
# One copy of every package, shared by all envs: the uv cache lives on the volume and the
# venvs hold SYMLINKS into it. The volume quota counts every hardlink as a full copy, and the
# worker envs share most of their bytes (torch 2.8 + the CUDA wheels, ~6 GB, in sep, tts and
# tts2): copied venvs counted 33.5 GB, symlinked ones count the cache once.
# Never run `uv cache clean` or `uv cache prune` on this cache: the venvs point into it.
export UV_CACHE_DIR=${DUB_UV_CACHE:-$ROOT/uv-cache} UV_PYTHON_INSTALL_DIR=$ROOT/python
export UV_LINK_MODE=symlink
export PATH=$ROOT/bin:$PATH
LAYOUT=symlink-v1  # part of the stamp: venvs from another layout are rebuilt

STAMP="$( (cat "$REQ" "$REPO/pyproject.toml"; echo "$LAYOUT") | sha256sum | cut -c1-16)"
if [ -f "$VENV/.dub-stamp" ] && [ "$(cat "$VENV/.dub-stamp")" = "$STAMP" ]; then
  uv pip install --quiet --python "$VENV/bin/python" --no-deps -e "$REPO"
  echo "[$NAME] up to date"
  exit 0
fi
echo "[$NAME] installing $(grep -cv '^\s*#' "$REQ" || true) requirement lines"
# a fresh venv every time: no stale packages, and with symlinks a rebuild is cheap
rm -rf "$VENV"
uv venv --quiet --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$REQ"
uv pip install --quiet --python "$VENV/bin/python" -e "$REPO"
echo "$STAMP" > "$VENV/.dub-stamp"
echo "[$NAME] done"
