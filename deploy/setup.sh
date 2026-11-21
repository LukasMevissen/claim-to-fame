#!/usr/bin/env bash
# Run ON THE SERVER inside ~/claim-to-fame after the repo, data/cases/ and .env are copied there.
# Creates the venv, runs the tests, installs and starts the systemd user service.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "missing .env (TEAM_API_KEY, OPENAI_KEY)"; exit 1; }
[ "$(ls data/cases/case_*.zip 2>/dev/null | wc -l)" -ge 101 ] || { echo "data/cases incomplete"; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version' || { echo "need Python >= 3.11"; exit 1; }
python3 -m venv .venv
.venv/bin/pip install -q -e ".[dev]"
.venv/bin/python -m pytest -q
timedatectl show -p NTPSynchronized --value 2>/dev/null | grep -q yes || echo "WARNING: clock not NTP-synced; key timing depends on it"
mkdir -p ~/.config/systemd/user
cp deploy/c2f.service ~/.config/systemd/user/c2f.service
loginctl enable-linger "$USER" 2>/dev/null || echo "WARNING: could not enable linger; service stops when you log out (use tmux instead)"
systemctl --user daemon-reload
systemctl --user enable --now c2f.service
echo "running. logs: journalctl --user -u c2f -f    per-game results: tail -f runs/submissions.jsonl"
