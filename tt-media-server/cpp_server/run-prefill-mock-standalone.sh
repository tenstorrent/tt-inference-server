#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

export LLM_MODE=prefill
export LLM_DEVICE_BACKEND=prefill
export TT_IPC_SHM_C2P=tt_ipc_c2p
export TT_IPC_SHM_P2C=tt_ipc_p2c
export TT_LOG_LEVEL=debug

SERVER_PID=""
RUNNER_PID=""
MOONCAKE_MASTER_PID=""

terminate_process_group() {
  local pid="$1"
  local name="$2"

  if [[ -z "$pid" ]]; then
    return
  fi

  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -- "-$pid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 -- "-$pid" 2>/dev/null; then
        return
      fi
      sleep 0.1
    done

    echo "$name did not exit after SIGTERM; sending SIGKILL." >&2
    kill -KILL -- "-$pid" 2>/dev/null || true
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  trap - EXIT INT TERM
  terminate_process_group "$SERVER_PID" "Server"
  terminate_process_group "$RUNNER_PID" "Runner"
  terminate_process_group "$MOONCAKE_MASTER_PID" "Mooncake master"
  wait "$SERVER_PID" 2>/dev/null || true
  wait "$RUNNER_PID" 2>/dev/null || true
  wait "$MOONCAKE_MASTER_PID" 2>/dev/null || true
}

on_interrupt() {
  cleanup
  exit 130
}

on_terminate() {
  cleanup
  exit 143
}

trap cleanup EXIT
trap on_interrupt INT
trap on_terminate TERM

setsid mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080 \
  > mooncake-master.log 2>&1 &
MOONCAKE_MASTER_PID=$!

setsid ./build/tt_media_server_cpp > prefill-server.log 2>&1 &
SERVER_PID=$!

setsid python3 src/runtime/runners/mock_prefill_runner.py > prefill-model.log 2>&1 &
RUNNER_PID=$!

echo "Mooncake master PID: $MOONCAKE_MASTER_PID"
echo "Server PID: $SERVER_PID"
echo "Runner PID: $RUNNER_PID"
echo "Logs: mooncake-master.log, prefill-server.log, prefill-model.log"

set +e
wait -n "$MOONCAKE_MASTER_PID" "$SERVER_PID" "$RUNNER_PID"
EXIT_CODE=$?
set -e

echo "One prefill process exited; stopping the others."
exit "$EXIT_CODE"