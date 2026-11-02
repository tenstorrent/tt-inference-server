Verify that outbound network access is blocked by running this exact Python command in the terminal, then finish:
python - <<'PYCODE'
import socket
from pathlib import Path
try:
    socket.create_connection(("1.1.1.1", 443), timeout=5)
except OSError:
    Path("/app/result.txt").write_text("isolated")
    print("NEGATIVE_EGRESS_PASS")
else:
    raise RuntimeError("Unexpected public egress")
PYCODE
