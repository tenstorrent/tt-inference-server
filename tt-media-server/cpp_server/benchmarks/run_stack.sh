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
# Optional decode-orchestrated Dynamo prefill:
#   DYNAMO_DECODE_ORCHESTRATES_PREFILL=1 DYNAMO_NATIVE_PREFILL_HANDOFF_ENABLED=1 ./run_stack.sh up
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
ROUTER_MODE="${ROUTER_MODE:-kv}"
SERVER_PORT="${SERVER_PORT:-8001}"       # decode/regular REST + dynamo endpoint
PREFILL_PORT="${PREFILL_PORT:-8002}"     # prefill REST
PREFILL_REPLICAS="${PREFILL_REPLICAS:-1}"
SOCKET_PORT="${SOCKET_PORT:-9000}"       # decode<->prefill inter-server socket
MAX_TOKENS_TO_PREFILL_ON_DECODE="${MAX_TOKENS_TO_PREFILL_ON_DECODE:-1000}"
DYNAMO_DECODE_ORCHESTRATES_PREFILL="${DYNAMO_DECODE_ORCHESTRATES_PREFILL:-0}"
DYNAMO_NATIVE_PREFILL_HANDOFF_ENABLED="${DYNAMO_NATIVE_PREFILL_HANDOFF_ENABLED:-0}"
DYNAMO_PREFILL_CLIENT_COMPONENT="${DYNAMO_PREFILL_CLIENT_COMPONENT:-tt-prefill}"
DYNAMO_PREFILL_CLIENT_NAMESPACE="${DYNAMO_PREFILL_CLIENT_NAMESPACE:-tt-prefill-routing}"
DYNAMO_PREFILL_ROUTER_ENABLED="${DYNAMO_PREFILL_ROUTER_ENABLED:-${DYNAMO_DECODE_ORCHESTRATES_PREFILL}}"
DYNAMO_PREFILL_ROUTER_COMPONENT="${DYNAMO_PREFILL_ROUTER_COMPONENT:-router}"
DYNAMO_PREFILL_ROUTER_ENDPOINT="${DYNAMO_PREFILL_ROUTER_ENDPOINT:-best_worker_id}"
DYNAMO_PREFILL_ROUTER_FALLBACK="${DYNAMO_PREFILL_ROUTER_FALLBACK:-error}"
DYNAMO_PREFILL_CLIENT_TIMEOUT_MS="${DYNAMO_PREFILL_CLIENT_TIMEOUT_MS:-60000}"
KV_CACHE_BLOCK_SIZE="${KV_CACHE_BLOCK_SIZE:-32}"
LLM_DEVICE_BACKEND="${LLM_DEVICE_BACKEND:-mock}"
ETCD_NAME="${ETCD_NAME:-etcd}"
PIDFILE="/tmp/tt_stack.pids"

FRONTEND_LOG="/tmp/tt_frontend.log"
DECODE_LOG="/tmp/tt_decode.log"
PREFILL_LOG="/tmp/tt_prefill.log"
ROUTER_LOG="/tmp/tt_router.log"

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
        local c; c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$c" in *"${BIN}"*|*"-m dynamo.frontend"*|*"-m dynamo.router"*) pids="$pids $p" ;; esac
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

# dynamo registration env for the decode worker (the one the frontend
# discovers).
worker_dynamo_env() {
    echo "DYNAMO_ENDPOINT_ENABLED=1"
    echo "DYNAMO_WORKER_ROLE=decode"
    echo "DYNAMO_DISCOVERY_BACKEND=etcd"
    echo "DYNAMO_ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
    echo "DYNAMO_NAMESPACE=default"
    echo "DYNAMO_COMPONENT=backend"
    echo "DYNAMO_ENDPOINT_NAME=generate"
    echo "DYNAMO_NATIVE_PREFILL_HANDOFF_ENABLED=${DYNAMO_NATIVE_PREFILL_HANDOFF_ENABLED}"
    echo "DYNAMO_DECODE_ORCHESTRATES_PREFILL=${DYNAMO_DECODE_ORCHESTRATES_PREFILL}"
    if [[ "${DYNAMO_DECODE_ORCHESTRATES_PREFILL}" == "1" ]]; then
        echo "DYNAMO_PREFILL_CLIENT_NAMESPACE=${DYNAMO_PREFILL_CLIENT_NAMESPACE}"
        echo "DYNAMO_PREFILL_CLIENT_COMPONENT=${DYNAMO_PREFILL_CLIENT_COMPONENT}"
        echo "DYNAMO_PREFILL_ROUTER_ENABLED=${DYNAMO_PREFILL_ROUTER_ENABLED}"
        echo "DYNAMO_PREFILL_ROUTER_COMPONENT=${DYNAMO_PREFILL_ROUTER_COMPONENT}"
        echo "DYNAMO_PREFILL_ROUTER_ENDPOINT=${DYNAMO_PREFILL_ROUTER_ENDPOINT}"
        echo "DYNAMO_PREFILL_ROUTER_FALLBACK=${DYNAMO_PREFILL_ROUTER_FALLBACK}"
        echo "DYNAMO_PREFILL_CLIENT_TIMEOUT_MS=${DYNAMO_PREFILL_CLIENT_TIMEOUT_MS}"
    fi
}

prefill_dynamo_env() {
    echo "DYNAMO_ENDPOINT_ENABLED=1"
    echo "DYNAMO_WORKER_ROLE=prefill"
    echo "DYNAMO_DISCOVERY_BACKEND=etcd"
    echo "DYNAMO_ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
    if [[ "${DYNAMO_DECODE_ORCHESTRATES_PREFILL}" == "1" ]]; then
        echo "DYNAMO_NAMESPACE=${DYNAMO_PREFILL_CLIENT_NAMESPACE}"
        echo "DYNAMO_COMPONENT=${DYNAMO_PREFILL_CLIENT_COMPONENT}"
    else
        echo "DYNAMO_NAMESPACE=default"
        echo "DYNAMO_COMPONENT=prefill"
    fi
    echo "DYNAMO_ENDPOINT_NAME=generate"
    echo "DYNAMO_NATIVE_PREFILL_HANDOFF_ENABLED=${DYNAMO_NATIVE_PREFILL_HANDOFF_ENABLED}"
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
            --router-mode "${ROUTER_MODE}" \
            --model-name "${MODEL_NAME}" \
            --model-path "${CPP_DIR}/tokenizers/${MODEL}" \
        > "${FRONTEND_LOG}" 2>&1 < /dev/null &
    echo $! >> "${PIDFILE}"
}

start_router() {
    [[ -x "${DYN_VENV}/bin/python3" ]] || die "dynamo venv not found at ${DYN_VENV}"
    log "router -> ${ROUTER_LOG} (endpoint ${DYNAMO_PREFILL_CLIENT_NAMESPACE}.${DYNAMO_PREFILL_CLIENT_COMPONENT}.generate)"
    setsid nohup env \
        DYN_DISCOVERY_BACKEND=etcd ETCD_ENDPOINTS="${ETCD_ENDPOINTS}" \
        DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq \
        "${DYN_VENV}/bin/python3" -m dynamo.router \
            --endpoint "${DYNAMO_PREFILL_CLIENT_NAMESPACE}.${DYNAMO_PREFILL_CLIENT_COMPONENT}.generate" \
            --router-block-size "${KV_CACHE_BLOCK_SIZE}" \
            --no-router-track-active-blocks \
        > "${ROUTER_LOG}" 2>&1 < /dev/null &
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

    log "decode :${SERVER_PORT} socket :${SOCKET_PORT}, prefill :${PREFILL_PORT} replicas=${PREFILL_REPLICAS}"
    start_worker "${DECODE_LOG}" "${SERVER_PORT}" \
        $(worker_dynamo_env) \
        LLM_MODE=decode LLM_DEVICE_BACKEND="${LLM_DEVICE_BACKEND}" \
        SOCKET_TRANSPORT=tcp SOCKET_HOST=0.0.0.0 SOCKET_PORT="${SOCKET_PORT}" \
        MAX_TOKENS_TO_PREFILL_ON_DECODE="${MAX_TOKENS_TO_PREFILL_ON_DECODE}"
    sleep 3
    # Distinct memory-queue + metrics shm names: decode and prefill are
    # co-located on one host, and these default to fixed names
    # (tt_mem_requests/_results, /tt_worker_metrics) — sharing them makes the
    # prefill worker's KV allocation requests race the decode worker's and hang.
    for replica in $(seq 0 $((PREFILL_REPLICAS - 1))); do
        replica_port=$((PREFILL_PORT + replica))
        replica_log="${PREFILL_LOG}"
        if [[ "${PREFILL_REPLICAS}" != "1" ]]; then
            replica_log="/tmp/tt_prefill_${replica}.log"
        fi
        prefill_env=()
        if [[ "${DYNAMO_DECODE_ORCHESTRATES_PREFILL}" == "1" ]]; then
            while IFS= read -r line; do prefill_env+=("${line}"); done < <(prefill_dynamo_env)
        fi
        start_worker "${replica_log}" "${replica_port}" \
            "${prefill_env[@]}" \
            LLM_MODE=prefill LLM_DEVICE_BACKEND="${LLM_DEVICE_BACKEND}" \
            SOCKET_TRANSPORT=tcp SOCKET_HOST=127.0.0.1 SOCKET_PORT="${SOCKET_PORT}" \
            TT_TASK_QUEUE="tt_tasks_prefill_${replica}" \
            TT_RESULT_QUEUE="tt_results_prefill_${replica}" \
            TT_CANCEL_QUEUE="tt_cancels_prefill_${replica}" \
            TT_WARMUP_SIGNALS_QUEUE="tt_warmup_signals_prefill_${replica}" \
            TT_MEMORY_REQUEST_QUEUE="tt_mem_requests_prefill_${replica}" \
            TT_MEMORY_RESULT_QUEUE="tt_mem_results_prefill_${replica}" \
            TT_WORKER_METRICS_SHM="/tt_worker_metrics_prefill_${replica}"
    done
    sleep 2
    if [[ "${DYNAMO_DECODE_ORCHESTRATES_PREFILL}" == "1" && "${DYNAMO_PREFILL_ROUTER_ENABLED}" == "1" ]]; then
        start_router
        sleep 2
    fi
    start_frontend
    wait_ready
    log "decode log: ${DECODE_LOG}  prefill log: ${PREFILL_LOG}"
    [[ -f "${ROUTER_LOG}" ]] && log "router log: ${ROUTER_LOG}"
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
