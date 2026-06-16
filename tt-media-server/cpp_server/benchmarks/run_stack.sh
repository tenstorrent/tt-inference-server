#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# run_stack.sh — bring up the Dynamo frontend + cpp_server (mock) for the
# prefill/decode test suite, without any tt-shield image. Frontend runs from a
# host ai-dynamo venv; etcd runs as the public quay image; workers run as the
# locally-built mock binary. See benchmarks/test_prefill_decode.py.
#
#   ./run_stack.sh up                      # disaggregated: decode + prefill split
#   ./run_stack.sh down                    # tear everything down
#
# Logs -> /tmp/tt_decode.log + /tmp/tt_prefill.log ; frontend -> /tmp/tt_frontend.log.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${CPP_DIR}/../.." && pwd)"
BIN="${CPP_DIR}/build/tt_media_server_cpp"

LOG_PREFIX="run_stack"
source "${REPO_DIR}/scripts/lib_dynamo_stack.sh"
DYN_VENV="${DYN_VENV:-${REPO_DIR}/dynamo-mock-backend/.venv}"

export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"

MODEL="${MODEL:-moonshotai/Kimi-K2.6}"
MODEL_NAME="${MODEL_NAME:-tt-cpp-server}"
HTTP_PORT="${HTTP_PORT:-8080}"
SERVER_PORT="${SERVER_PORT:-8001}"       # decode/regular REST + dynamo endpoint
PREFILL_PORT="${PREFILL_PORT:-8002}"     # prefill REST
SOCKET_PORT="${SOCKET_PORT:-9000}"       # decode<->prefill inter-server socket
MAX_TOKENS_TO_PREFILL_ON_DECODE="${MAX_TOKENS_TO_PREFILL_ON_DECODE:-1000}"
ETCD_NAME="${ETCD_NAME:-etcd}"
PIDFILE="/tmp/tt_stack.pids"

FRONTEND_LOG="/tmp/tt_frontend.log"
DECODE_LOG="/tmp/tt_decode.log"
PREFILL_LOG="/tmp/tt_prefill.log"

teardown() {
    log "tearing down"
    set +e                            # cleanup is best-effort; never abort on a dead pid
    local pids=""
    [[ -f "${PIDFILE}" ]] && pids="$(tr '\n' ' ' < "${PIDFILE}")"
    # also sweep stray workers/frontend (orphaned --worker children, prior runs)
    for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        [[ "$p" == "$$" ]] && continue
        local c; c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$c" in *"${BIN}"*|*"-m dynamo.frontend"*) pids="$pids $p" ;; esac
    done
    pids="$(echo "${pids}" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')"
    if [[ -n "${pids// }" ]]; then
        kill ${pids} 2>/dev/null || true
        for _ in $(seq 1 12); do      # wait for graceful exit, else SIGKILL
            local alive=""
            for p in ${pids}; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
            [[ -z "${alive// }" ]] && break
            sleep 0.5
        done
        for p in ${pids}; do kill -9 "$p" 2>/dev/null || true; done
    fi
    rm -f "${PIDFILE}"
    # Worker IPC shm (boost/POSIX) is kernel-persistent and survives SIGKILL; a stale
    # segment makes a fresh worker attach to a corrupt queue and hang. Remove the
    # lowercase tt_* segments (NOT the uppercase TT_UMD_LOCK.* tt-metal locks).
    rm -f /dev/shm/tt_* 2>/dev/null || true
    # block until our ports are actually released (avoids relaunch bind races)
    for _ in $(seq 1 30); do
        ss -ltn 2>/dev/null | grep -qE ":(${HTTP_PORT}|${SERVER_PORT}|${PREFILL_PORT}|${SOCKET_PORT})[[:space:]]" || break
        sleep 0.5
    done
    set -e
    log "down"
}

ensure_etcd() {
    if ! docker ps --format '{{.Names}}' | grep -qx "${ETCD_NAME}"; then
        log "starting etcd"
        docker rm -f "${ETCD_NAME}" >/dev/null 2>&1 || true
        start_etcd "${ETCD_NAME}"
    fi
    wait_etcd_healthy "${ETCD_NAME}"
    ETCD_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${ETCD_NAME}")
    [[ -n "${ETCD_IP}" ]] || die "could not resolve etcd container IP"
    ETCD_ENDPOINTS="http://${ETCD_IP}:2379"
    log "etcd at ${ETCD_ENDPOINTS}"
}

start_frontend() {
    [[ -x "${DYN_VENV}/bin/python3" ]] || die "dynamo venv not found at ${DYN_VENV}"
    log "frontend -> ${FRONTEND_LOG} (http :${HTTP_PORT})"
    setsid nohup env \
        DYN_DISCOVERY_BACKEND=etcd ETCD_ENDPOINTS="${ETCD_ENDPOINTS}" \
        DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq \
        DYN_TOKENIZER="${DYN_TOKENIZER:-fastokens}" \
        "${DYN_VENV}/bin/python3" -m dynamo.frontend \
            --http-port "${HTTP_PORT}" \
            --model-name "${MODEL_NAME}" \
            --model-path "${CPP_DIR}/tokenizers/${MODEL}" \
        > "${FRONTEND_LOG}" 2>&1 < /dev/null &
    echo $! >> "${PIDFILE}"
}

# start_worker <logfile> <rest-port> <env line>...
start_worker() {
    local logf="$1" port="$2"; shift 2
    setsid nohup env "$@" TT_LOG_LEVEL="${TT_LOG_LEVEL:-debug}" \
        MODEL="${MODEL}" MAX_SESSIONS_COUNT=128 \
        "${BIN}" -p "${port}" > "${logf}" 2>&1 < /dev/null &
    echo $! >> "${PIDFILE}"
}

wait_ready() {
    log "waiting for /v1/models to list ${MODEL}"
    for i in $(seq 1 40); do
        if curl -s "http://127.0.0.1:${HTTP_PORT}/v1/models" 2>/dev/null | grep -q "${MODEL}"; then
            log "ready after ${i}s"; return 0
        fi
        sleep 1
    done
    die "frontend never listed ${MODEL}; see ${FRONTEND_LOG} and worker logs"
}

up() {
    [[ -x "${BIN}" ]] || { log "no binary; building (mock)"; (cd "${CPP_DIR}" && env -u TT_METAL_HOME ./build.sh); }
    teardown
    : > "${PIDFILE}"
    ensure_etcd

    log "decode :${SERVER_PORT} socket :${SOCKET_PORT}, prefill :${PREFILL_PORT}"
    # The decode worker registers with the frontend; the prefill worker does NOT.
    start_worker "${DECODE_LOG}" "${SERVER_PORT}" \
        $(dynamo_worker_env "${ETCD_ENDPOINTS}") \
        LLM_MODE=decode LLM_DEVICE_BACKEND=mock \
        SOCKET_TRANSPORT=tcp SOCKET_HOST=0.0.0.0 SOCKET_PORT="${SOCKET_PORT}" \
        MAX_TOKENS_TO_PREFILL_ON_DECODE="${MAX_TOKENS_TO_PREFILL_ON_DECODE}"
    sleep 3
    # Distinct memory-queue + metrics shm names: decode and prefill are
    # co-located on one host, and these default to fixed names
    # (tt_mem_requests/_results, /tt_worker_metrics) — sharing them makes the
    # prefill worker's KV allocation requests race the decode worker's and hang.
    start_worker "${PREFILL_LOG}" "${PREFILL_PORT}" \
        LLM_MODE=prefill LLM_DEVICE_BACKEND=mock \
        SOCKET_TRANSPORT=tcp SOCKET_HOST=127.0.0.1 SOCKET_PORT="${SOCKET_PORT}" \
        TT_MEMORY_REQUEST_QUEUE=tt_mem_requests_prefill \
        TT_MEMORY_RESULT_QUEUE=tt_mem_results_prefill \
        TT_WORKER_METRICS_SHM=/tt_worker_metrics_prefill
    sleep 2
    start_frontend
    wait_ready
    log "decode log: ${DECODE_LOG}  prefill log: ${PREFILL_LOG}"
    log "frontend: http://127.0.0.1:${HTTP_PORT}  (model id: ${MODEL})"
    log "pids: $(tr '\n' ' ' < "${PIDFILE}")"
}

[[ $# -eq 1 ]] || { echo "usage: $0 up | down" >&2; exit 1; }
CMD="$1"

case "${CMD}" in
    up)   up ;;
    down) teardown ;;
    *)    die "unknown command: ${CMD}" ;;
esac
