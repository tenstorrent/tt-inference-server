#!/bin/sh
mkdir -p /logs/verifier
python - <<'PYCODE'
from pathlib import Path
ok = Path('/app/result.txt').exists() and Path('/app/result.txt').read_text().strip() == '42'
Path('/logs/verifier/reward.txt').write_text('1' if ok else '0')
PYCODE
