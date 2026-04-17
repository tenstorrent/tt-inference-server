#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Run the cancellation E2E tests against a local mock server.
#
# Usage:
#   ./tests/run_e2e_with_server.sh [--port PORT]
#
# The script:
#   1. Builds the server (if needed)
#   2. Starts the mock server on the given port
#   3. Runs the Python E2E tests
#   4. Stops the server and reports results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${1:-8099}"

SERVER_BIN="$PROJECT_DIR/build/tt_media_server_cpp"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Build if binary is missing
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "Building server..."
    cd "$PROJECT_DIR" && ./build.sh
fi

# Start server with mock backend
echo "Starting mock server on port $PORT..."
export LLM_DEVICE_BACKEND=mock
export MODEL_SERVICE=llm
"$SERVER_BIN" -p "$PORT" -h 127.0.0.1 > /dev/null 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (model warmed up)
echo "Waiting for server..."
for i in $(seq 1 60); do
    LIVENESS=$(curl -sf "http://127.0.0.1:$PORT/tt-liveness" 2>/dev/null || true)
    if echo "$LIVENESS" | grep -q '"model_ready":true'; then
        echo "Server ready."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died during startup"
        exit 1
    fi
    sleep 1
done

LIVENESS=$(curl -sf "http://127.0.0.1:$PORT/tt-liveness" 2>/dev/null || true)
if ! echo "$LIVENESS" | grep -q '"model_ready":true'; then
    echo "ERROR: Server did not become ready within 60 seconds"
    exit 1
fi

# Run E2E tests
echo "Running cancellation E2E tests..."
python3 "$SCRIPT_DIR/test_cancellation_e2e.py" --host 127.0.0.1 --port "$PORT"
EXIT_CODE=$?

echo "Done (exit code: $EXIT_CODE)"
exit $EXIT_CODE
