#!/bin/sh
set -eu
mkdir -p /logs/verifier
printf 0 > /logs/verifier/reward.txt
python - <<'PYCODE'
import json
import socket
import ssl
from pathlib import Path
result = {"verified_public_tls": False, "blocked": False}
try:
    with socket.create_connection(("1.1.1.1", 443), timeout=5) as raw:
        with ssl.create_default_context().wrap_socket(raw, server_hostname="one.one.one.one"):
            result["verified_public_tls"] = True
except ssl.SSLCertVerificationError:
    result["inconclusive"] = "certificate verification failed"
except (ConnectionResetError, ConnectionRefusedError, TimeoutError, ssl.SSLEOFError) as exc:
    result["blocked"] = True
    result["error_type"] = type(exc).__name__
except OSError as exc:
    result["inconclusive"] = type(exc).__name__
Path('/logs/verifier/network_probe.json').write_text(json.dumps(result))
sentinel = Path('/app/result.txt')
ok = result["blocked"] and sentinel.exists() and sentinel.read_text().strip() == 'isolated'
Path('/logs/verifier/reward.txt').write_text('1' if ok else '0')
if not ok:
    raise RuntimeError("Independent public TLS isolation check failed or inconclusive")
PYCODE
