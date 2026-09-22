#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run bash setup.sh first.' >&2
    exit 1
fi
# A user service runs as the SSH account, including when nobody is logged in
# after `sudo loginctl enable-linger USER`. No shell expansion in the unit.
.venv/bin/python - <<'PY'
from pathlib import Path

root = Path.cwd()
if any(character in str(root) for character in ('\n', '\r')):
    raise SystemExit('Move the repository to a path without line breaks before installing the service.')
def unit_quote(value):
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'

unit = f'''[Unit]
Description=Neuroplex artificial life simulation
After=network.target

[Service]
Type=simple
WorkingDirectory={str(root).replace('%', '%%')}
ExecStart={unit_quote(root / '.venv/bin/python')} -m neuroplex --host 127.0.0.1 --port 8000 --data-dir {unit_quote(root / 'data')}
Environment=OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
NoNewPrivileges=true
UMask=0077

[Install]
WantedBy=default.target
'''
directory = Path.home() / '.config/systemd/user'
directory.mkdir(parents=True, exist_ok=True)
target = directory / 'neuroplex.service'
target.write_text(unit)
print(f'Installed {target}')
PY
systemctl --user daemon-reload
systemctl --user enable --now neuroplex.service
echo 'Service started. Check: systemctl --user status neuroplex'
echo 'For start at boot, run once: sudo loginctl enable-linger "$(whoami)"'
