#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Starts a mock C++ server, runs the Python e2e tests, and tears down.
# Intended to be called via ctest so that `ctest --output-on-failure` runs
# both C++ unit tests and Python e2e tests in a single invocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
SERVER_BIN="${BUILD_DIR}/tt_media_server_cpp"

# Check Python dependencies; skip (exit 0) if missing so ctest doesn't fail.
if ! command -v pytest >/dev/null 2>&1; then
  # Fall back to python3 -m pytest
  if ! python3 -c "import pytest, requests" 2>/dev/null; then
    echo "SKIP: pytest/requests not available; install with: pip install requests pytest" >&2
    exit 0
  fi
  PYTEST="python3 -m pytest"
else
  if ! python3 -c "import requests" 2>/dev/null; then
    echo "SKIP: python requests not available; install with: pip install requests" >&2
    exit 0
  fi
  PYTEST="pytest"
fi
PORT="${E2E_PORT:-18765}"
PID_FILE="/tmp/tt_e2e_server_${PORT}.pid"
LOG_FILE="/tmp/tt_e2e_server_${PORT}.log"

cleanup() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "ERROR: Server binary not found at $SERVER_BIN" >&2
  echo "Build the project first with ./build.sh" >&2
  exit 1
fi

# Start mock server
LLM_DEVICE_BACKEND=mock "$SERVER_BIN" -p "$PORT" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Wait for health
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${PORT}/health" | grep -q healthy; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Server process died during startup. Log:" >&2
    cat "$LOG_FILE" >&2
    exit 1
  fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:${PORT}/health" | grep -q healthy; then
  echo "ERROR: Server did not become healthy within 30s. Log:" >&2
  cat "$LOG_FILE" >&2
  exit 1
fi

echo "Server ready on port $PORT (PID $SERVER_PID)"

# Run e2e tests
SERVER_BASE_URL="http://127.0.0.1:${PORT}" \
  $PYTEST "${SCRIPT_DIR}/test_cancellation_e2e.py" -v --tb=short
