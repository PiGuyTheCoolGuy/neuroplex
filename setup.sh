#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit('Neuroplex needs Python 3.12 or newer (Ubuntu 24.04 includes it).')
PY
if ! python3 -m venv .venv; then
    echo 'Install prerequisites: sudo apt install python3-venv python3-pip' >&2
    exit 1
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
echo 'Setup complete. Start with: bash start.sh'
