#!/usr/bin/env bash
# Bootstraps and verifies the Day 0A environment on a clean checkout.
# Usage: bash setup.sh
set -euo pipefail

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q
