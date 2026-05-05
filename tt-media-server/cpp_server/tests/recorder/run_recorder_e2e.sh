#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Drive the runner-event recorder against a local mock server.
#
# Usage:
#   ./tests/recorder/run_recorder_e2e.sh record   # record fresh fixture
#   ./tests/recorder/run_recorder_e2e.sh assert   # verify against committed fixture
#
# Optional env vars:
#   RECORDER_PORT    -- port to bind the test server on (default 8099)
#   RECORDER_FIXTURE -- path to JSONL fixture (default tests/recorder/expected/baseline.jsonl)
#
# The script:
#   1. Builds the server (if missing).
#   2. Starts a single-worker mock server with TT_RUNNER_RECORDER_MODE set.
#   3. Runs `scenarios.py` to drive a curated request set.
#   4. Stops the server and forwards its exit code.
#
# In `assert` mode, the server returns non-zero if any recorded event
# diverges from the fixture, so this script exits non-zero too.

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "record" && "$MODE" != "assert" ]]; then
    echo "Usage: $0 record|assert" >&2
    exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PORT="${RECORDER_PORT:-8099}"
FIXTURE="${RECORDER_FIXTURE:-$SCRIPT_DIR/expected/baseline.jsonl}"
SERVER_BIN="$PROJECT_DIR/build/tt_media_server_cpp"

mkdir -p "$(dirname "$FIXTURE")"

if [[ ! -x "$SERVER_BIN" ]]; then
    echo "[recorder] Building server..."
    (cd "$PROJECT_DIR" && ./build.sh)
fi

if [[ "$MODE" == "assert" && ! -f "$FIXTURE" ]]; then
    echo "[recorder] ERROR: fixture not found at $FIXTURE." >&2
    echo "[recorder] Run \`$0 record\` once and commit the file." >&2
    exit 65
fi

# Single-worker, mock backend, deterministic request ordering.
export TT_RUNNER_RECORDER_MODE="$MODE"
export TT_RUNNER_RECORDER_PATH="$FIXTURE"
export LLM_DEVICE_BACKEND=mock
export MODEL_SERVICE=llm
export NUM_WORKERS=1

LOG_DIR="$PROJECT_DIR/build/recorder_logs"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server-$MODE.log"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[recorder] Stopping server (PID $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        # Wait up to 10s for finalize() + clean shutdown.
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

echo "[recorder] mode=$MODE port=$PORT fixture=$FIXTURE"
echo "[recorder] server log: $SERVER_LOG"

"$SERVER_BIN" -p "$PORT" -h 127.0.0.1 -t 4 >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

python3 "$SCRIPT_DIR/scenarios.py" --host 127.0.0.1 --port "$PORT"
SCENARIOS_EXIT=$?

# Trigger a clean shutdown via SIGTERM so the server's main() can call
# RunnerEventRecorder::finalize() (which sets the exit code in assert mode).
echo "[recorder] Sending SIGTERM to server..."
kill -TERM "$SERVER_PID" 2>/dev/null || true

# Capture the real exit code without `set -e` masking via `if !` wrapper.
set +e
wait "$SERVER_PID"
SERVER_EXIT=$?
set -e
SERVER_PID=""

echo "[recorder] scenarios_exit=$SCENARIOS_EXIT server_exit=$SERVER_EXIT"

if [[ $SCENARIOS_EXIT -ne 0 ]]; then
    exit $SCENARIOS_EXIT
fi
exit $SERVER_EXIT
