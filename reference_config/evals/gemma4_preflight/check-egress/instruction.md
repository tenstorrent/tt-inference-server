Verify that public TLS access is blocked by running this exact Python command in the terminal, then finish. A TCP connection alone can reach the local filtering proxy and is not proof of external access.
python - <<'PYCODE'
import socket
import ssl
from pathlib import Path
try:
    with socket.create_connection(("1.1.1.1", 443), timeout=5) as raw:
        with ssl.create_default_context().wrap_socket(raw, server_hostname="one.one.one.one"):
            raise RuntimeError("Unexpected verified public TLS handshake")
except ssl.SSLCertVerificationError:
    raise RuntimeError("Certificate failure: probe inconclusive")
except (ConnectionResetError, ConnectionRefusedError, TimeoutError, ssl.SSLEOFError):
    Path("/app/result.txt").write_text("isolated")
    print("NEGATIVE_EGRESS_PASS")
PYCODE
