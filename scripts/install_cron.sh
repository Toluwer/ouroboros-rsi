#!/usr/bin/env bash
# Registers (or refreshes) a crontab entry that runs one improvement cycle
# every 30 minutes. Cycles run fully offline once the dataset and base model
# are cached; commits accumulate locally and are pushed whenever
# GITHUB_TOKEN is set in the environment (add it to a secure env source).
#
# Use this when you want the system improving on a schedule on your own
# machine instead of as a long-running daemon.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
PYBIN="$(command -v $PY)"

ENTRY="*/30 * * * * cd $REPO && $PYBIN -m ouroboros cycle >> workspace/logs/cron.log 2>&1 # ouroboros-rsi"

mkdir -p "$REPO/workspace/logs"
(crontab -l 2>/dev/null | grep -v "ouroboros-rsi" || true; echo "$ENTRY") | crontab -

echo "[install_cron] registered: one cycle every 30 minutes"
crontab -l | grep ouroboros
