#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Run cancellation E2E tests through the Dynamo frontend.
#
# Prerequisites (same as MainIntegration / DisaggregatedE2E):
#   - etcd + Dynamo frontend already running, e.g.
#       cd dynamo_frontend && ./deploy.sh --no-monitoring --no-worker
#   - or test-gate's etcd + dynamo.frontend steps
#
# This script:
#   1. Starts a mock cpp_server worker that registers DynamoWorkerServer
#   2. Waits until the frontend lists the model
#   3. Runs cancellation_e2e_test.py against the frontend
#   4. Stops the worker (SIGTERM, then SIGKILL if needed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DYNAMO_HOST="${DYNAMO_HOST:-127.0.0.1}"
DYNAMO_PORT="${DYNAMO_PORT:-8080}"
DYNAMO_MODEL="${DYNAMO_MODEL:-deepseek-ai/DeepSeek-R1-0528}"
ETCD_ENDPOINTS="${DYNAMO_ETCD_ENDPOINTS:-${ETCD_ENDPOINTS:-http://127.0.0.1:2379}}"
# Worker still binds Drogon (unused for chat); pick a free-ish port for -p.
WORKER_HTTP_PORT="${WORKER_HTTP_PORT:-8099}"

SERVER_BIN="$PROJECT_DIR/build/tt_media_server_cpp"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping worker (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        # DynamoWorkerServer::stop() can block; don't hang ctest.
        for _ in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.25
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Worker still alive; sending SIGKILL"
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ ! -x "$SERVER_BIN" ]]; then
    echo "Building server..."
    cd "$PROJECT_DIR" && ./build.sh --blaze
fi

# Fail fast if frontend is not up (test-gate / deploy.sh --no-worker).
if ! curl -sf --max-time 2 "http://${DYNAMO_HOST}:${DYNAMO_PORT}/v1/models" >/dev/null; then
    echo "ERROR: Dynamo frontend not reachable at http://${DYNAMO_HOST}:${DYNAMO_PORT}"
    echo "Start with: cd dynamo_frontend && ./deploy.sh --no-monitoring --no-worker"
    echo "(test-gate already starts etcd + frontend before ctest)"
    exit 1
fi

echo "Starting mock Dynamo worker (Drogon :${WORKER_HTTP_PORT}, discovery via etcd)..."
export LLM_DEVICE_BACKEND=mock
export MODEL_SERVICE=llm
export DYNAMO_ENDPOINT_ENABLED=1
export DYNAMO_DISCOVERY_BACKEND="${DYNAMO_DISCOVERY_BACKEND:-etcd}"
export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-etcd}"
export DYNAMO_ETCD_ENDPOINTS="$ETCD_ENDPOINTS"
export ETCD_ENDPOINTS="$ETCD_ENDPOINTS"
export DYNAMO_NAMESPACE="${DYNAMO_NAMESPACE:-default}"
export DYNAMO_COMPONENT="${DYNAMO_COMPONENT:-backend}"
export DYNAMO_ENDPOINT_NAME="${DYNAMO_ENDPOINT_NAME:-generate}"
export DYNAMO_BIND_HOST="${DYNAMO_BIND_HOST:-0.0.0.0}"
export DYN_TCP_RPC_HOST="${DYN_TCP_RPC_HOST:-127.0.0.1}"
export MODEL_NAME="${MODEL_NAME:-$DYNAMO_MODEL}"

"$SERVER_BIN" -p "$WORKER_HTTP_PORT" -h 127.0.0.1 -t 4 > /tmp/cancellation_e2e_worker.log 2>&1 &
SERVER_PID=$!

echo "Waiting for worker to register (model visible in /v1/models)..."
READY=0
for _ in $(seq 1 90); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Worker process died during startup"
        tail -n 80 /tmp/cancellation_e2e_worker.log || true
        exit 1
    fi
    if curl -sf --max-time 2 "http://${DYNAMO_HOST}:${DYNAMO_PORT}/v1/models" 2>/dev/null \
        | grep -q "$DYNAMO_MODEL"; then
        READY=1
        break
    fi
    sleep 1
done

if [[ "$READY" != "1" ]]; then
    echo "ERROR: Model ${DYNAMO_MODEL} not visible in /v1/models within 90s"
    tail -n 80 /tmp/cancellation_e2e_worker.log || true
    exit 1
fi
echo "Frontend lists ${DYNAMO_MODEL}; running cancellation E2E..."

python3 "$SCRIPT_DIR/../cancellation_e2e_test.py" \
    --host "$DYNAMO_HOST" \
    --port "$DYNAMO_PORT" \
    --model "$DYNAMO_MODEL"
EXIT_CODE=$?

echo "Done (exit code: $EXIT_CODE)"
exit $EXIT_CODE
