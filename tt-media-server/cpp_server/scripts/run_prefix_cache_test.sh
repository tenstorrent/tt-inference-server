#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# One-shot prefix-cache E2E test.
#
# Starts the cpp_server, waits for readiness, runs the prefix-cache test
# suite against it, then tears everything down.
#
# By default tests hit the cpp_server's HTTP endpoint directly (port 8000),
# which exercises the same LLMPipeline / SessionManager / prefix-cache code
# as the full Dynamo path.  To test through the Dynamo frontend (requires
# etcd), use: USE_DYNAMO=1 ./scripts/run_prefix_cache_test.sh
#
# Usage:
#   ./scripts/run_prefix_cache_test.sh
#   TEST_VERBOSE=1 ./scripts/run_prefix_cache_test.sh
#   USE_DYNAMO=1 ./scripts/run_prefix_cache_test.sh   # full Dynamo stack
#
# Env vars:
#   SERVER_PORT         cpp_server HTTP port   (default 8000)
#   USE_DYNAMO          Set to 1 to start full Dynamo frontend + etcd path
#   HTTP_PORT           Dynamo frontend port   (default 9000, only with USE_DYNAMO)
#   MODEL_NAME          Model name             (default tt-cpp-server)
#   TEST_VERBOSE        Set to 1 for verbose test output
#   TEST_TIMEOUT        Seconds to wait for server readiness (default 90)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${CPP_DIR}/build}"
BIN="${BUILD_DIR}/tt_media_server_cpp"

SERVER_PORT="${SERVER_PORT:-8000}"
HTTP_PORT="${HTTP_PORT:-9000}"
MODEL_NAME="${MODEL_NAME:-tt-cpp-server}"
TEST_TIMEOUT="${TEST_TIMEOUT:-90}"
TEST_VERBOSE="${TEST_VERBOSE:-0}"
USE_DYNAMO="${USE_DYNAMO:-0}"

BACKEND_PID=""
STACK_PID=""

cleanup() {
    echo ""
    echo "Tearing down…"
    [[ -n "${STACK_PID}" ]]   && { kill -- -"${STACK_PID}" 2>/dev/null || kill "${STACK_PID}" 2>/dev/null || true; }
    [[ -n "${BACKEND_PID}" ]] && kill "${BACKEND_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- build if needed ----------------------------------------------------------

if [[ ! -x "${BIN}" ]]; then
    echo "Building cpp_server…"
    "${CPP_DIR}/build.sh"
fi

# --- start stack --------------------------------------------------------------

if [[ "${USE_DYNAMO}" == "1" ]]; then
    echo "=== Prefix-cache E2E test (Dynamo frontend) ==="
    echo "  Frontend : http://127.0.0.1:${HTTP_PORT}"
    echo "  Backend  : http://127.0.0.1:${SERVER_PORT}"
    echo ""
    export HTTP_PORT SERVER_PORT MODEL_NAME
    "${SCRIPT_DIR}/start_dynamo.sh" &
    STACK_PID=$!
    TEST_PORT="${HTTP_PORT}"
else
    echo "=== Prefix-cache E2E test (cpp_server direct) ==="
    echo "  cpp_server : http://127.0.0.1:${SERVER_PORT}"
    echo ""
    "${BIN}" -p "${SERVER_PORT}" &
    BACKEND_PID=$!
    TEST_PORT="${SERVER_PORT}"
fi

# --- wait for readiness -------------------------------------------------------

echo "Waiting for server on :${TEST_PORT}…"
DEADLINE=$((SECONDS + TEST_TIMEOUT))
READY=0
while (( SECONDS < DEADLINE )); do
    if curl -sf "http://127.0.0.1:${TEST_PORT}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

if (( READY == 0 )); then
    echo "ERROR: Server not ready within ${TEST_TIMEOUT}s"
    exit 1
fi
echo "Server ready."

# --- run tests ----------------------------------------------------------------

VERBOSE_FLAG=""
if [[ "${TEST_VERBOSE}" == "1" ]]; then
    VERBOSE_FLAG="--verbose"
fi

echo ""
echo "Running prefix-cache tests…"
echo ""

python3 "${CPP_DIR}/tests/test_prefix_cache_e2e.py" \
    --port "${TEST_PORT}" \
    --timeout 30 \
    ${VERBOSE_FLAG}

EXIT_CODE=$?

echo ""
if (( EXIT_CODE == 0 )); then
    echo "All prefix-cache tests passed."
else
    echo "Some prefix-cache tests failed (exit ${EXIT_CODE})."
fi

exit "${EXIT_CODE}"
