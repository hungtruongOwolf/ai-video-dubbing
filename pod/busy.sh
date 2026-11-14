#!/bin/bash
# Run a command on the pod without dubd's idle watchdog terminating the pod meanwhile:
# the watchdog counts jobs, worker calls, and this heartbeat file touched in the last 2 min.
#   bash /workspace/dub/repo/pod/busy.sh curl -o big.tar.gz https://...
BUSY_FILE="${DUB_BUSY_FILE:-/tmp/dub.busy}"
( while true; do touch "$BUSY_FILE"; sleep 30; done ) &
hb=$!
trap 'kill $hb 2>/dev/null' EXIT
"$@"
