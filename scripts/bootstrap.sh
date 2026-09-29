#!/usr/bin/env bash
# One-command bootstrap: install dependencies, run operator phase-0
# fine-tuning, then hand over to the autonomous daemon.
#
#   GITHUB_TOKEN=ghp_... ./scripts/bootstrap.sh
#
# Set OUROBOROS_DAEMON=0 to stop after phase 0.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

echo "[bootstrap] installing dependencies (CPU torch)"
$PY -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
$PY -m pip install --quiet -r requirements.txt

if [ ! -f workspace/state.json ]; then
    echo "[bootstrap] phase 0: operator fine-tuning (required before autonomy)"
    $PY -m ouroboros phase0
else
    echo "[bootstrap] state found; skipping phase 0"
fi

if [ "${OUROBOROS_DAEMON:-1}" = "1" ]; then
    echo "[bootstrap] starting autonomous daemon"
    exec $PY -m ouroboros daemon
fi
