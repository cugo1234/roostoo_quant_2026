#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3 python3-venv git
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
mkdir -p logs data results
printf '\nBootstrap complete. Copy .env.example to .env, add credentials, keep LIVE_TRADING=0 for dry-run, then run tests.\n'
