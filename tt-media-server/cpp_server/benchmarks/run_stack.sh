#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# run_stack.sh — bring up the Dynamo frontend + cpp_server (mock_pipeline) for the
# prefill/decode test suite, without any tt-shield image. Frontend runs from a
# host ai-dynamo venv; etcd runs as the public quay image; workers run as the
# locally-built mock_pipeline binary. See benchmarks/test_prefill_decode.py.
#
#   ./run_stack.sh up                      # direct cpp_server socket split
#   DYNAMO_NATIVE_ROUTING=1 ./run_stack.sh up
#                                          # native Dynamo decode/prefill split
#   ./run_stack.sh down                    # tear everything down
#
# Logs -> /tmp/tt_decode.log + /tmp/tt_prefill.log ; frontend -> /tmp/tt_frontend.log.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${CPP_DIR}/../.." && pwd)"
BIN="${CPP_DIR}/build/tt_media_server_cpp"
DYN_VENV="${DYN_VENV:-${REPO_DIR}/dynamo-mock-backend/.venv}"

export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"

MODEL="${MODEL:-moonshotai/Kimi-K2.6}"
MODEL_NAME="${MODEL_NAME:-tt-cpp-server}"
HTTP_PORT="${HTTP_PORT:-8080}"
SERVER_PORT="${SERVER_PORT:-8001}"       # decode/regular REST + dynamo endpoint
PREFILL_PORT="${PREFILL_PORT:-8002}"     # prefill REST
SOCKET_PORT="${SOCKET_PORT:-9000}"       # decode<->prefill inter-server socket
MAX_TOKENS_TO_PREFILL_ON_DECODE="${MAX_TOKENS_TO_PREFILL_ON_DECODE:-1000}"
DYNAMO_NATIVE_ROUTING="${DYNAMO_NATIVE_ROUTING:-0}"
DYNAMO_NATIVE_NAMESPACE="${DYNAMO_NATIVE_NAMESPACE:-dynamo}"
ETCD_NAME="${ETCD_NAME:-etcd}"
PIDFILE="/tmp/tt_stack.pids"

FRONTEND_LOG="/tmp/tt_frontend.log"
DECODE_LOG="/tmp/tt_decode.log"
PREFILL_LOG="/tmp/tt_prefill.log"

log() { printf '[run_stack] %s\n' "$*"; }
die() { printf '[run_stack] %s\n' "$*" >&2; exit 1; }

teardown() {
    log "tearing down"
    set +e                            # cleanup is best-effort; never abort on a dead pid
    local pids=""
    [[ -f "${PIDFILE}" ]] && pids="$(tr '\n' ' ' < "${PIDFILE}")"
    # also sweep stray workers/frontend (orphaned --worker children, prior runs)
    for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        [[ "$p" == "$$" ]] && continue
        [[ -r "/proc/$p/cmdline" ]] || continue
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
        docker run -d --name "${ETCD_NAME}" -p 2379:2379 quay.io/coreos/etcd:v3.5.13 \
            /usr/local/bin/etcd --name dyn-etcd \
                --advertise-client-urls http://0.0.0.0:2379 \
                --listen-client-urls http://0.0.0.0:2379 >/dev/null
        sleep 3
    fi
    ETCD_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${ETCD_NAME}")
    [[ -n "${ETCD_IP}" ]] || die "could not resolve etcd container IP"
    docker exec "${ETCD_NAME}" etcdctl endpoint health >/dev/null 2>&1 || die "etcd unhealthy"
    ETCD_ENDPOINTS="http://${ETCD_IP}:2379"
    log "etcd at ${ETCD_ENDPOINTS}"
}

# Dynamo registration env. Direct mode registers only the decode worker under
# default/backend/generate. Native mode registers decode and prefill workers as
# separate Dynamo worker types under dynamo/{decode,prefill}/generate.
worker_dynamo_env() {
    local namespace="${1:-default}"
    local component="${2:-backend}"
    local worker_type="${3:-}"
    local model_type="${4-Chat}"
    echo "DYNAMO_ENDPOINT_ENABLED=1"
    echo "DYNAMO_DISCOVERY_BACKEND=etcd"
    echo "DYNAMO_ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
    echo "DYNAMO_NAMESPACE=${namespace}"
    echo "DYNAMO_COMPONENT=${component}"
    echo "DYNAMO_ENDPOINT_NAME=generate"
    [[ -n "${model_type}" ]] && echo "DYNAMO_MODEL_TYPE=${model_type}"
    [[ -n "${worker_type}" ]] && echo "DYNAMO_WORKER_TYPE=${worker_type}"
}

worker_ipc_env() {
    local ns="$1"
    echo "TT_WARMUP_SIGNALS_QUEUE=tt_warmup_signals_${ns}"
    echo "TT_TASK_QUEUE=tt_tasks_${ns}"
    echo "TT_RESULT_QUEUE=tt_results_${ns}"
    echo "TT_CANCEL_QUEUE=tt_cancels_${ns}"
    echo "TT_MEMORY_REQUEST_QUEUE=tt_mem_requests_${ns}"
    echo "TT_MEMORY_RESULT_QUEUE=tt_mem_results_${ns}"
    echo "TT_WORKER_METRICS_SHM=/tt_worker_metrics_${ns}"
}

start_frontend() {
    [[ -x "${DYN_VENV}/bin/python3" ]] || die "dynamo venv not found at ${DYN_VENV}"
    "${DYN_VENV}/bin/python3" -c 'import dynamo.frontend' >/dev/null 2>&1 \
        || die "dynamo.frontend not importable from ${DYN_VENV}; install ai-dynamo there or set DYN_VENV to a Dynamo venv"
    log "frontend -> ${FRONTEND_LOG} (http :${HTTP_PORT})"
    setsid nohup env \
        DYN_DISCOVERY_BACKEND=etcd ETCD_ENDPOINTS="${ETCD_ENDPOINTS}" \
        DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq \
        DYN_REQUEST_PLANE_CODEC="${DYN_REQUEST_PLANE_CODEC:-json}" \
        DYN_TOKENIZER="${DYN_TOKENIZER:-fastokens}" \
        "${DYN_VENV}/bin/python3" -m dynamo.frontend \
            --http-port "${HTTP_PORT}" \
            --router-mode "${ROUTER_MODE:-kv}" \
            --model-name "${MODEL_NAME}" \
            --model-path "${CPP_DIR}/tokenizers/${MODEL}" \
        > "${FRONTEND_LOG}" 2>&1 < /dev/null &
    echo $! >> "${PIDFILE}"
}

ensure_binary() {
    local rebuild=0
    if [[ ! -x "${BIN}" ]]; then
        log "no binary; building with --blaze for mock_pipeline"
        rebuild=1
    elif [[ ! -f "${CPP_DIR}/build/CMakeCache.txt" ]] || \
         ! grep -q '^ENABLE_BLAZE:BOOL=ON$' "${CPP_DIR}/build/CMakeCache.txt"; then
        log "existing binary was not built with --blaze; rebuilding for mock_pipeline"
        rebuild=1
    else
        local src
        for src in \
            "${CPP_DIR}/src/dynamo/dynamo_endpoint.cpp" \
            "${CPP_DIR}/src/dynamo/discovery.cpp" \
            "${CPP_DIR}/src/services/disaggregation_service.cpp" \
            "${CPP_DIR}/src/services/llm_pipeline.cpp" \
            "${CPP_DIR}/src/utils/service_factory.cpp" \
            "${CPP_DIR}/src/main.cpp" \
            "${CPP_DIR}/include/dynamo/dynamo_endpoint.hpp" \
            "${CPP_DIR}/include/dynamo/discovery.hpp" \
            "${CPP_DIR}/include/services/disaggregation_service.hpp" \
            "${CPP_DIR}/include/services/llm_pipeline.hpp"; do
            if [[ "${src}" -nt "${BIN}" ]]; then
                log "source newer than binary: ${src#$CPP_DIR/}"
                rebuild=1
                break
            fi
        done
    fi

    if [[ "${rebuild}" == "1" ]]; then
        (cd "${CPP_DIR}" && env -u TT_METAL_HOME ./build.sh --blaze)
    fi
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
            log "/v1/models ready after ${i}s"
            break
        fi
        sleep 1
        if [[ "${i}" == "40" ]]; then
            die "frontend never listed ${MODEL}; see ${FRONTEND_LOG} and worker logs"
        fi
    done

    log "waiting for /v1/chat/completions route"
    for i in $(seq 1 40); do
        local status
        status="$(curl -s -o /dev/null -w "%{http_code}" \
            -H 'Content-Type: application/json' \
            -d '{}' \
            "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" 2>/dev/null || true)"
        if [[ "${status}" != "000" && "${status}" != "404" ]]; then
            log "chat completions route ready after ${i}s"
            break
        fi
        sleep 1
        if [[ "${i}" == "40" ]]; then
            die "frontend chat completions route did not become ready; see ${FRONTEND_LOG}"
        fi
    done

    if [[ "${DYNAMO_NATIVE_ROUTING}" == "1" ]]; then
        log "waiting for native Dynamo router activation"
        for i in $(seq 1 40); do
            if grep -aq "Prefill router activated successfully" "${FRONTEND_LOG}"; then
                log "native router ready after ${i}s"
                return 0
            fi
            sleep 1
        done
        die "native Dynamo router did not finish activation; see ${FRONTEND_LOG}"
    fi

    log "ready"
}

wait_worker_healthy() {
    local name="$1" port="$2" logf="$3"
    log "waiting for ${name} worker health on :${port}"
    for i in $(seq 1 40); do
        local body
        body="$(curl -s "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
        if [[ "${body}" == *'"status":"healthy"'* ]]; then
            log "${name} healthy after ${i}s"
            return 0
        fi
        if [[ "${body}" == *'"no workers are alive"'* ]]; then
            die "${name} has no live workers; see ${logf}"
        fi
        sleep 1
    done
    die "${name} did not become healthy; see ${logf}"
}

up() {
    ensure_binary
    teardown
    : > "${PIDFILE}"
    ensure_etcd

    if [[ "${DYNAMO_NATIVE_ROUTING}" == "1" ]]; then
        log "native Dynamo routing: decode :${SERVER_PORT}, prefill :${PREFILL_PORT}"
        start_worker "${DECODE_LOG}" "${SERVER_PORT}" \
            $(worker_dynamo_env "${DYNAMO_NATIVE_NAMESPACE}" decode decode Chat) \
            $(worker_ipc_env native_decode) \
            LLM_MODE=decode LLM_DEVICE_BACKEND=mock_pipeline \
            USE_PREFILL_GATEWAY=0 DYNAMO_NATIVE_ROUTING=1
        wait_worker_healthy "decode" "${SERVER_PORT}" "${DECODE_LOG}"
        start_worker "${PREFILL_LOG}" "${PREFILL_PORT}" \
            $(worker_dynamo_env "${DYNAMO_NATIVE_NAMESPACE}" prefill prefill Prefill) \
            $(worker_ipc_env native_prefill) \
            LLM_MODE=prefill LLM_DEVICE_BACKEND=mock_pipeline \
            USE_PREFILL_GATEWAY=0 DYNAMO_NATIVE_ROUTING=1 \
            DYNAMO_MODEL_INPUT=Tokens
        wait_worker_healthy "prefill" "${PREFILL_PORT}" "${PREFILL_LOG}"
    else
        log "direct cpp_server socket routing: decode :${SERVER_PORT} socket :${SOCKET_PORT}, prefill :${PREFILL_PORT}"
        start_worker "${DECODE_LOG}" "${SERVER_PORT}" \
            $(worker_dynamo_env) \
            $(worker_ipc_env direct_decode) \
            LLM_MODE=decode LLM_DEVICE_BACKEND=mock_pipeline \
            SOCKET_HOST=0.0.0.0 SOCKET_PORT="${SOCKET_PORT}" \
            MAX_TOKENS_TO_PREFILL_ON_DECODE="${MAX_TOKENS_TO_PREFILL_ON_DECODE}"
        wait_worker_healthy "decode" "${SERVER_PORT}" "${DECODE_LOG}"
        start_worker "${PREFILL_LOG}" "${PREFILL_PORT}" \
            $(worker_ipc_env direct_prefill) \
            LLM_MODE=prefill LLM_DEVICE_BACKEND=mock_pipeline \
            SOCKET_HOST=127.0.0.1 SOCKET_PORT="${SOCKET_PORT}"
        wait_worker_healthy "prefill" "${PREFILL_PORT}" "${PREFILL_LOG}"
    fi
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
