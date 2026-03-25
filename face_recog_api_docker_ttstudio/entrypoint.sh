#!/usr/bin/env bash
set -euo pipefail

# Use absolute paths (not $HOME) so --user root still works
APP_USER_HOME="/home/container_app_user"
TT_METAL_DIR="${APP_USER_HOME}/tt-metal"
FACE_API_DIR="${APP_USER_HOME}/face-recognition-api"

# tt-metal must be on PYTHONPATH and we must run FROM tt-metal dir
# so relative paths like "models/demos/yunet/YUNet/weights/best.pt" resolve.
export PYTHONPATH="${TT_METAL_DIR}:${FACE_API_DIR}:${PYTHONPATH:-}"

cd "${TT_METAL_DIR}"

if [[ -x "${TT_METAL_DIR}/python_env/bin/python" ]]; then
  PY="${TT_METAL_DIR}/python_env/bin/python"
else
  PY="python3"
fi

exec "${PY}" -m uvicorn app.main:app --host 0.0.0.0 --port "${SERVICE_PORT:-7070}"
