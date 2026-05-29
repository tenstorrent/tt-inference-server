#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Launch the Dynamo frontend together with cpp_server's Dynamo TCP backend.
#
# Required venv layout:
#   ${DYN_VENV} must contain `bin/python3 -m dynamo.frontend` (set up via the
#   sibling dynamo-mock-backend `setup.sh` or any Dynamo install).
#
# Usage:
#   ./scripts/start_dynamo.sh                              # default file store
#   DYN_DISCOVERY_BACKEND=etcd ./scripts/start_dynamo.sh   # use etcd
#   DYNAMO_DISCOVERY_PATH=/tmp/dyn HTTP_PORT=9000 ./scripts/start_dynamo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
BIN="${BUILD_DIR}/tt_media_server_cpp"

# --- venv -------------------------------------------------------------------
# Default to dynamo-mock-backend's venv if it exists; users can override with
# DYN_VENV.
DEFAULT_VENV="${SCRIPT_DIR}/../../../dynamo-mock-backend/.venv"
DYN_VENV="${DYN_VENV:-${DEFAULT_VENV}}"
if [[ ! -d "${DYN_VENV}" ]]; then
    echo "Dynamo venv not found at ${DYN_VENV}." >&2
    echo "Set DYN_VENV to a directory containing 'bin/python3 -m dynamo.frontend'." >&2
    exit 1
fi

# --- discovery + endpoint ---------------------------------------------------
# DYN_DISCOVERY_BACKEND: "file" (default) or "etcd".
#   - file: writes JSON under DYN_FILE_STORE; both processes mount the same path.
#   - etcd: requires a reachable etcd at ETCD_ENDPOINTS (default
#           http://localhost:2379). The frontend reads the same env var.
export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_REQUEST_PLANE="${DYN_REQUEST_PLANE:-tcp}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"
# HTTP request plane (when DYN_REQUEST_PLANE=http). Must match on frontend + worker.
export DYN_HTTP_RPC_ROOT_PATH="${DYN_HTTP_RPC_ROOT_PATH:-/v1/rpc}"
export DYN_HTTP_RPC_PORT="${DYN_HTTP_RPC_PORT:-8888}"
export DYNAMO_REQUEST_PLANE="${DYN_REQUEST_PLANE}"
export DYN_FILE_STORE="${DYN_FILE_STORE:-/tmp/dynamo_store_kv}"
export ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://localhost:2379}"
export DYNAMO_ENDPOINT_ENABLED=1
# Mirror the Dynamo-side env into cpp_server's namespace so the C++ accessors
# (which key off DYNAMO_*) see the same values without duplicating exports.
export DYNAMO_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND}"
export DYNAMO_DISCOVERY_PATH="${DYNAMO_DISCOVERY_PATH:-${DYN_FILE_STORE}}"
export DYNAMO_ETCD_ENDPOINTS="${DYNAMO_ETCD_ENDPOINTS:-${ETCD_ENDPOINTS}}"
export DYNAMO_NAMESPACE="${DYNAMO_NAMESPACE:-default}"
export DYNAMO_COMPONENT="${DYNAMO_COMPONENT:-backend}"
export DYNAMO_ENDPOINT_NAME="${DYNAMO_ENDPOINT_NAME:-generate}"

# Quick sanity check: when running with etcd, fail fast if the gateway isn't
# reachable so we don't waste a minute booting cpp_server only to crash on
# first lease grant.
if [[ "${DYN_DISCOVERY_BACKEND}" == "etcd" ]]; then
    ETCD_HOST_PORT="${ETCD_ENDPOINTS#http://}"
    ETCD_HOST_PORT="${ETCD_HOST_PORT%%/*}"
    if ! (echo > "/dev/tcp/${ETCD_HOST_PORT/:/\/}") 2>/dev/null; then
        echo "etcd not reachable at ${ETCD_ENDPOINTS}." >&2
        echo "Start one with:  docker run --rm -d -p 2379:2379 --name etcd quay.io/coreos/etcd:v3.5.13 \\" >&2
        echo "    /usr/local/bin/etcd --advertise-client-urls=http://0.0.0.0:2379 --listen-client-urls=http://0.0.0.0:2379" >&2
        exit 1
    fi
fi

# Resolve the model path the cpp_server's tokenizers/ tree exposes for the
# active backend so the frontend ends up tokenizing against the same files
# the backend uses. Mirrors `tokenizerDirForModel` in src/utils/tokenizers/
# tokenizer.cpp. Falls back to user-supplied MODEL_PATH.
LLM_BACKEND="${LLM_DEVICE_BACKEND:-mock_pipeline}"
case "${LLM_BACKEND}" in
    llama*|*llama*) MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct" ;;
    *)              MODEL_DIR="deepseek-ai/DeepSeek-R1-0528" ;;
esac
DEFAULT_MODEL_PATH="${SCRIPT_DIR}/tokenizers/${MODEL_DIR}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

# --- build ------------------------------------------------------------------
if [[ ! -x "${BIN}" ]]; then
    echo "Building cpp_server..."
    "${SCRIPT_DIR}/build.sh"
fi

# --- launch -----------------------------------------------------------------
HTTP_PORT="${HTTP_PORT:-9000}"
SERVER_PORT="${SERVER_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-tt-cpp-server}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dynamo-bypass}"

if [[ "${DYN_DISCOVERY_BACKEND}" == "file" ]]; then
    rm -rf "${DYN_FILE_STORE}"
    DISCOVERY_LABEL="file://${DYN_FILE_STORE}"
else
    DISCOVERY_LABEL="${DYN_DISCOVERY_BACKEND}://${ETCD_ENDPOINTS}"
fi

cleanup() {
    echo "" >&2
    echo "Shutting down..." >&2
    [[ -n "${FRONTEND_PID:-}" ]] && kill "${FRONTEND_PID}" 2>/dev/null || true
    [[ -n "${BACKEND_PID:-}" ]] && kill "${BACKEND_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Dynamo + cpp_server ==="
echo "  Frontend HTTP : ${HTTP_PORT}"
echo "  cpp_server    : http://0.0.0.0:${SERVER_PORT}"
echo "  Discovery     : ${DISCOVERY_LABEL}"
echo "  Model name    : ${MODEL_NAME}"
echo "  Model path    : ${MODEL_PATH}"
echo "  LLM backend   : ${LLM_BACKEND}"
echo ""

echo "Starting Dynamo frontend..."
"${DYN_VENV}/bin/python3" -m dynamo.frontend \
    --http-port "${HTTP_PORT}" \
    --model-name "${MODEL_NAME}" \
    --model-path "${MODEL_PATH}" &
FRONTEND_PID=$!

sleep 2

echo "Starting cpp_server (DYNAMO_ENDPOINT_ENABLED=1)..."
"${BIN}" -p "${SERVER_PORT}" &
BACKEND_PID=$!

wait "${BACKEND_PID}"
