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
export UV_CACHE_DIR=$ROOT/uv-cache UV_PYTHON_INSTALL_DIR=$ROOT/python UV_LINK_MODE=hardlink
export PATH=$ROOT/bin:$PATH

STAMP="$(cat "$REQ" "$REPO/pyproject.toml" | sha256sum | cut -c1-16)"
if [ -f "$VENV/.dub-stamp" ] && [ "$(cat "$VENV/.dub-stamp")" = "$STAMP" ]; then
  uv pip install --quiet --python "$VENV/bin/python" --no-deps -e "$REPO"
  echo "[$NAME] up to date"
  exit 0
fi
echo "[$NAME] installing $(grep -cv '^\s*#' "$REQ" || true) requirement lines"
[ -x "$VENV/bin/python" ] || uv venv --quiet --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$REQ"
uv pip install --quiet --python "$VENV/bin/python" -e "$REPO"
echo "$STAMP" > "$VENV/.dub-stamp"
echo "[$NAME] done"
