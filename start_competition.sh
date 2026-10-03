#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example to .env and add your Roostoo credentials." >&2
  exit 1
fi

set -a
source ./.env
set +a
export PYTHONPATH=src

if [[ "${ROOSTOO_ENV:-}" != "competition" ]]; then
  echo "ROOSTOO_ENV must be competition" >&2
  exit 1
fi
if [[ "${LIVE_TRADING:-0}" != "1" ]]; then
  echo "LIVE_TRADING is not 1. Set it to 1 in .env when you are ready." >&2
  exit 1
fi
if [[ "${ROOSTOO_LIVE_TRADING_CONFIRM:-}" != "YES" ]]; then
  echo "ROOSTOO_LIVE_TRADING_CONFIRM must be YES" >&2
  exit 1
fi

exec .venv/bin/python scripts/run_live.py
