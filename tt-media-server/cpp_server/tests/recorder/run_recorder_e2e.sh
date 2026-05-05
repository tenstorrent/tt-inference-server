#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Boot a single-worker mock server with the runner-event recorder enabled
# and run `test_runner_events.py` against it. The test asserts inline what
# the API+service layer hands to the worker queue (no fixture files).
#
# Optional env vars:
#   RECORDER_PORT -- port to bind the test server on (default 8099)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PORT="${RECORDER_PORT:-8099}"
SERVER_BIN="$PROJECT_DIR/build/tt_media_server_cpp"

if [[ ! -x "$SERVER_BIN" ]]; then
    echo "[recorder] Building server..."
    (cd "$PROJECT_DIR" && ./build.sh)
fi

# Recorder is opt-in via env; the debug HTTP endpoints return 404 unless
# this is set so the surface is invisible in production.
export TT_RUNNER_RECORDER_ENABLED=1
export LLM_DEVICE_BACKEND=mock
export MODEL_SERVICE=llm
export NUM_WORKERS=1

LOG_DIR="$PROJECT_DIR/build/recorder_logs"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[recorder] Stopping server (PID $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 100); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[recorder] port=$PORT"
echo "[recorder] server log: $SERVER_LOG"

"$SERVER_BIN" -p "$PORT" -h 127.0.0.1 -t 4 >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

set +e
python3 "$SCRIPT_DIR/test_runner_events.py" --host 127.0.0.1 --port "$PORT"
TEST_EXIT=$?
set -e

echo "[recorder] test_exit=$TEST_EXIT"
exit $TEST_EXIT
