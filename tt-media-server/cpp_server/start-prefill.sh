#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LLM_MODE=prefill
export LLM_DEVICE_BACKEND=prefill
export TT_IPC_SHM_C2P=tt_ipc_c2p
export TT_IPC_SHM_P2C=tt_ipc_p2c

SERVER_PID=""
RUNNER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
    kill "$RUNNER_PID" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  wait "$RUNNER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

./build/tt_media_server_cpp > prefill-server.log 2>&1 &
SERVER_PID=$!

python3 src/runtime/runners/mock_prefill_runner.py > prefill-model.log 2>&1 &
RUNNER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Runner PID: $RUNNER_PID"
echo "Logs: prefill-server.log, prefill-model.log"

set +e
wait -n "$SERVER_PID" "$RUNNER_PID"
EXIT_CODE=$?
set -e

echo "One prefill process exited; stopping the other."
exit "$EXIT_CODE"