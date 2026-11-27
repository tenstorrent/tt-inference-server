#!/usr/bin/env bash
# Run when tt_media_server_cpp is up and a Tenstorrent device can be opened.
# The contract cases are the same as with the mock runner (submit -> poll -> download). The
# extra check requires the finished job to download as a non-empty mp4 produced
# by the device runner. VIDEO_JOB_TIMEOUT (seconds, default 1800) bounds polling.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${VIDEO_BASE_URL:-http://127.0.0.1:8000}"
DEVICE="${VIDEO_DEVICE:-/dev/tenstorrent/0}"
TIMEOUT="${VIDEO_READY_TIMEOUT:-120}"
export VIDEO_API_KEY="${VIDEO_API_KEY:-your-secret-key}"

echo "server: $BASE"
echo "device: $DEVICE"
echo "waiting for model_ready for at most ${TIMEOUT}s"

if [[ ! -e "$DEVICE" ]]; then
  echo "$DEVICE does not exist. The device test stops before any HTTP request."
  exit 1
fi

deadline=$((SECONDS + TIMEOUT))
ready=0
while (( SECONDS < deadline )); do
  if python3 - "$BASE" <<'PY'
import json, sys, urllib.request
url = sys.argv[1].rstrip("/") + "/tt-liveness"
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        body = json.loads(resp.read().decode())
except Exception:
    sys.exit(1)
sys.exit(0 if body.get("model_ready") is True else 1)
PY
  then
    ready=1
    break
  fi
  sleep 2
done

if (( ready == 0 )); then
  echo "model_ready did not become true at $BASE/tt-liveness"
  echo "start the server from tt-media-server/cpp_server:"
  echo "  MODEL_SERVICE=video MODEL=Wan2.2-T2V-A14B-Diffusers MODEL_RUNNER=tt-wan2.2 \\"
  echo "  DEVICE=t3k DEVICE_IDS=\"(0,1,2,3)\" TT_METAL_HOME=<tt-metal> \\"
  echo "  ./build/tt_media_server_cpp -p 8000"
  exit 1
fi

exec python3 "$HERE/mock/contract_cases.py" --base-url "$BASE" --require-file
