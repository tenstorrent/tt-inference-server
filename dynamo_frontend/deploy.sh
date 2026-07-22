#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_PREFIX="deploy"
source "${REPO_ROOT}/scripts/lib_dynamo_stack.sh"

# Fixed names/ports (not CLI-configurable).
NETWORK_NAME="dynamo-net"
ETCD_NAME="etcd"
WORKER_NAME="tt-cpp-worker"
PREFILL_WORKER_NAME="tt-cpp-prefill-worker"
FRONTEND_NAME="dynamo-frontend"
FRONTEND_HOST_PORT="8080"
PROMETHEUS_HOST_PORT="${PROMETHEUS_HOST_PORT:-9090}"
GRAFANA_HOST_PORT="${GRAFANA_HOST_PORT:-3000}"
MODEL_NAME="tt-cpp-server"
LLM_DEVICE_BACKEND="${LLM_DEVICE_BACKEND:-pipeline_manager}"
MONITORING_COMPOSE="${REPO_ROOT}/tt-media-server/monitoring/docker-compose.yml"
MONITORING_PROJECT_NAME="dynamo-monitoring"
MONITORING_PROMETHEUS_NAME="dynamo-prometheus"
MONITORING_GRAFANA_NAME="dynamo-grafana"
MONITORING_PROCESS_EXPORTER_NAME="dynamo-process-exporter"
PREFILL_GATEWAY_NAME="prefill-gateway"
DEFAULT_PREFILL_GATEWAY_IMAGE="tt-prefill-gateway:dev"
PREFILL_GATEWAY_IMAGE="${PREFILL_GATEWAY_IMAGE:-$DEFAULT_PREFILL_GATEWAY_IMAGE}"
PREFILL_GATEWAY_DECODE_PORT="${PREFILL_GATEWAY_DECODE_PORT:-7100}"
PREFILL_GATEWAY_METRICS_PORT="${PREFILL_GATEWAY_METRICS_PORT:-9091}"
PREFILL_GATEWAY_HEALTH_PORT="${PREFILL_GATEWAY_HEALTH_PORT:-9092}"
PREFILL_GATEWAY_PREFILL_BIND="${PREFILL_GATEWAY_PREFILL_BIND:-0.0.0.0:7200}"
PREFILL_WORKER_PORT="${PREFILL_WORKER_PORT:-8001}"
PREFILL_DIRECT_SOCKET_PORT="${PREFILL_DIRECT_SOCKET_PORT:-9000}"

# Image defaults (override with the matching flag if needed).
# ETCD_IMAGE default comes from scripts/lib_dynamo_stack.sh.
WORKER_IMAGE="ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-blaze:ef76035_20260605_091948"
FRONTEND_IMAGE="ghcr.io/tenstorrent/tt-shield/tt-dynamo-frontend:ef76035_20260605_091917"

KIMI_MODEL_ID="moonshotai/Kimi-K2.6"
DEEPSEEK_MODEL_ID="deepseek-ai/DeepSeek-R1-0528"
HF_MODEL_ID="$DEEPSEEK_MODEL_ID"
DEVICE_IDS="10,11,14,15,18,19,22,23"

# --local-build mounts this repo's cpp_server/build over the worker image and
# runs the local binary; defaults to the in-repo cpp_server so no path is needed.
LOCAL_BUILD=""
CPP_SERVER_DIR="${SCRIPT_DIR}/../tt-media-server/cpp_server"
MONITORING_ENABLED=1
MONITORING_STARTED=0
PREFILL_GATEWAY_ENABLED=0
PREFILL_DIRECT_ENABLED=0
DYNAMO_ROUTING_ENABLED=0
PREFILL_GATEWAY_STARTED=0
PREFILL_WORKERS_STARTED=()
PREFILL_WORKER_COUNT="${PREFILL_WORKER_COUNT:-1}"

port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        [[ -n "$(ss -ltnH "sport = :${port}" 2>/dev/null)" ]] && return 0
    fi
    (echo >"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
}

require_host_port_free() {
    local port="$1"
    local label="$2"
    local override_env="${3:-}"
    if port_in_use "$port"; then
        if [[ -n "$override_env" ]]; then
            die "${label} host port ${port} is already in use. Set ${override_env}=<free-port> and rerun, or use --no-monitoring."
        fi
        die "${label} host port ${port} is already in use."
    fi
}

ensure_prefill_gateway_image() {
    if docker image inspect "$PREFILL_GATEWAY_IMAGE" >/dev/null 2>&1; then
        return
    fi

    if [[ "$PREFILL_GATEWAY_IMAGE" != "$DEFAULT_PREFILL_GATEWAY_IMAGE" ]]; then
        die "prefill gateway image not found: $PREFILL_GATEWAY_IMAGE"
    fi

    log "building prefill gateway image ($PREFILL_GATEWAY_IMAGE)"
    docker build \
        -f "${REPO_ROOT}/tt-media-server/Dockerfile.gateway" \
        -t "$PREFILL_GATEWAY_IMAGE" \
        "$REPO_ROOT" >/dev/null
}

endpoint_port() {
    local endpoint="$1"
    printf '%s' "${endpoint##*:}"
}

usage() {
    cat >&2 <<EOF
Usage: $0 [options]
  --kimi | --deepseek          model to serve (default: --deepseek)
  --hf-model-id <id>           explicit HF model id (overrides the above)
  --local-build                run this repo's cpp_server/build in the worker
  --etcd-image <img>           (default: ${ETCD_IMAGE})
  --worker-image <img>         (default: ${WORKER_IMAGE})
  --frontend-image <img>       (default: ${FRONTEND_IMAGE})
  --device-ids <ids>           cpp_server DEVICE_IDS (default: ${DEVICE_IDS})
  --llm-device-backend <name>   cpp_server LLM_DEVICE_BACKEND (default: ${LLM_DEVICE_BACKEND})
  --prefill-gateway            start PrefillGateway and route decode prefill requests through it
  --prefill-gateway-image <img> (default: ${PREFILL_GATEWAY_IMAGE}; auto-builds default if missing)
  --prefill-gateway-prefill-bind <host:port>
                               ZMQ prefill bind endpoint (default: ${PREFILL_GATEWAY_PREFILL_BIND})
  --prefill-workers <count>    managed prefill worker count for gateway/Dynamo routing modes (default: ${PREFILL_WORKER_COUNT})
  --prefill-direct             start one managed prefill worker connected directly to decode
  --dynamo-routing             EXPERIMENTAL: register decode/prefill pools and let Dynamo own routing decisions
  --no-monitoring              skip Prometheus + Grafana deployment

LLM_DEVICE_BACKEND, HF_TOKEN, and perf knobs (ROUTER_MODE, DYN_TOKENIZER,
RAYON_NUM_THREADS, DYN_RUNTIME_*, RUST_LOG, DYN_TX_TRACE,
DYN_ENABLE_ANTHROPIC_API) are read from the environment.

Monitoring is enabled by default and uses tt-media-server/monitoring/docker-compose.yml.
Override SERVER_TARGET/SERVER_SERVICE/GATEWAY_TARGET/GF_HOME_DASHBOARD in the
environment if you want Prometheus to scrape a different container or dashboard.
Set PROMETHEUS_HOST_PORT/GRAFANA_HOST_PORT if 9090/3000 are already in use.
Prefill gateway knobs can also be set through PREFILL_GATEWAY_* environment variables.
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kimi)           HF_MODEL_ID="$KIMI_MODEL_ID";     shift ;;
        --deepseek)       HF_MODEL_ID="$DEEPSEEK_MODEL_ID"; shift ;;
        --hf-model-id)    HF_MODEL_ID="$2";                 shift 2 ;;
        --local-build)    LOCAL_BUILD=1;                    shift ;;
        --etcd-image)     ETCD_IMAGE="$2";                  shift 2 ;;
        --worker-image)   WORKER_IMAGE="$2";                shift 2 ;;
        --frontend-image) FRONTEND_IMAGE="$2";              shift 2 ;;
        --device-ids)     DEVICE_IDS="$2";                  shift 2 ;;
        --llm-device-backend) LLM_DEVICE_BACKEND="$2";      shift 2 ;;
        --prefill-gateway) PREFILL_GATEWAY_ENABLED=1;       shift ;;
        --prefill-gateway-image) PREFILL_GATEWAY_IMAGE="$2"; shift 2 ;;
        --prefill-gateway-prefill-bind)
            PREFILL_GATEWAY_ENABLED=1
            PREFILL_GATEWAY_PREFILL_BIND="$2"
            shift 2
            ;;
        --prefill-workers) PREFILL_WORKER_COUNT="$2"; shift 2 ;;
        --prefill-direct) PREFILL_DIRECT_ENABLED=1;       shift ;;
        --dynamo-routing) DYNAMO_ROUTING_ENABLED=1; shift ;;
        --no-monitoring)  MONITORING_ENABLED=0;             shift ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

[[ "$PREFILL_WORKER_COUNT" =~ ^[1-9][0-9]*$ ]] || die "--prefill-workers must be a positive integer"

# Local-build: validate the binary and prepare the bind-mount + entrypoint.
LOCAL_BUILD_MOUNT=()
WORKER_ENTRYPOINT=()
if [[ -n "$LOCAL_BUILD" ]]; then
    CPP_SERVER_DIR_ABS="$(readlink -f "$CPP_SERVER_DIR" 2>/dev/null || true)"
    [[ -d "$CPP_SERVER_DIR_ABS" ]] || die "cpp_server directory not found: $CPP_SERVER_DIR"
    [[ -f "$CPP_SERVER_DIR_ABS/build/tt_media_server_cpp" ]] \
        || die "no binary at $CPP_SERVER_DIR_ABS/build/tt_media_server_cpp — run ./build.sh --blaze first"
    log "using local build from $CPP_SERVER_DIR_ABS"
    LOCAL_BUILD_MOUNT+=(-v "${CPP_SERVER_DIR_ABS}/build:/home/container_app_user/app/server/cpp_server/build:ro")
    WORKER_ENTRYPOINT+=(--entrypoint /bin/bash)
fi

if [[ "$MONITORING_ENABLED" == "1" ]]; then
    [[ -f "$MONITORING_COMPOSE" ]] || die "monitoring compose file not found: $MONITORING_COMPOSE"
    docker compose version >/dev/null 2>&1 || die "docker compose is required for monitoring"
fi
ROUTING_MODE_COUNT=$((PREFILL_GATEWAY_ENABLED + PREFILL_DIRECT_ENABLED + DYNAMO_ROUTING_ENABLED))
[[ "$ROUTING_MODE_COUNT" -le 1 ]] || die "--prefill-direct, --prefill-gateway, and --dynamo-routing are mutually exclusive"
if [[ "$PREFILL_GATEWAY_ENABLED" == "1" ]]; then
    ensure_prefill_gateway_image
fi

require_host_port_free 2379 "etcd"
require_host_port_free "$FRONTEND_HOST_PORT" "Dynamo frontend"
if [[ "$MONITORING_ENABLED" == "1" ]]; then
    require_host_port_free "$PROMETHEUS_HOST_PORT" "Prometheus" "PROMETHEUS_HOST_PORT"
    require_host_port_free "$GRAFANA_HOST_PORT" "Grafana" "GRAFANA_HOST_PORT"
fi

cleanup() {
    log "tearing down"
    if [[ "${MONITORING_STARTED:-0}" == "1" ]]; then
        TT_NET="$NETWORK_NAME" \
            PROMETHEUS_CONTAINER_NAME="$MONITORING_PROMETHEUS_NAME" \
            GRAFANA_CONTAINER_NAME="$MONITORING_GRAFANA_NAME" \
            PROCESS_EXPORTER_CONTAINER_NAME="$MONITORING_PROCESS_EXPORTER_NAME" \
            PROMETHEUS_HOST_PORT="$PROMETHEUS_HOST_PORT" \
            GRAFANA_HOST_PORT="$GRAFANA_HOST_PORT" \
            docker compose -p "$MONITORING_PROJECT_NAME" -f "$MONITORING_COMPOSE" down >/dev/null 2>&1 || true
    fi
    if [[ "${PREFILL_GATEWAY_STARTED:-0}" == "1" ]]; then
        docker rm -f "$PREFILL_GATEWAY_NAME" >/dev/null 2>&1 || true
    fi
    if [[ "${#PREFILL_WORKERS_STARTED[@]}" -gt 0 ]]; then
        docker rm -f "${PREFILL_WORKERS_STARTED[@]}" >/dev/null 2>&1 || true
    fi
    docker rm -f "$FRONTEND_NAME" "$WORKER_NAME" "$ETCD_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker network inspect "$NETWORK_NAME" >/dev/null 2>&1 \
    || { log "creating network $NETWORK_NAME"; docker network create "$NETWORK_NAME" >/dev/null; }

# ── etcd ──────────────────────────────────────────────────────────────────
log "starting etcd ($ETCD_IMAGE)"
start_etcd "$ETCD_NAME" "$NETWORK_NAME"
log "waiting for etcd"
wait_etcd_healthy "$ETCD_NAME"
log "etcd healthy"

if [[ "$PREFILL_GATEWAY_ENABLED" == "1" ]]; then
    docker ps -a --format '{{.Names}}' | grep -q "^${PREFILL_GATEWAY_NAME}\$" \
        && die "container already exists: ${PREFILL_GATEWAY_NAME}"

    PREFILL_GATEWAY_ARGS=(
        --decode-port="$PREFILL_GATEWAY_DECODE_PORT"
        --metrics-port="$PREFILL_GATEWAY_METRICS_PORT"
        --health-port="$PREFILL_GATEWAY_HEALTH_PORT"
        --prefill-bind="$PREFILL_GATEWAY_PREFILL_BIND"
    )

    log "starting PrefillGateway ($PREFILL_GATEWAY_IMAGE)"
    docker run -d --name "$PREFILL_GATEWAY_NAME" --network "$NETWORK_NAME" \
        "$PREFILL_GATEWAY_IMAGE" \
        "${PREFILL_GATEWAY_ARGS[@]}" >/dev/null
    PREFILL_GATEWAY_STARTED=1

    sleep 2
    docker ps --format '{{.Names}}' | grep -q "^${PREFILL_GATEWAY_NAME}\$" \
        || { docker logs --tail 80 "$PREFILL_GATEWAY_NAME" >&2 || true; die "PrefillGateway exited during startup"; }
fi

# ── worker ────────────────────────────────────────────────────────────────
DEVICE_ARGS=()
[[ -e /dev/tenstorrent ]] && DEVICE_ARGS+=(--device /dev/tenstorrent --cap-add=SYS_NICE)

# Model-specific env for Blaze prefix and metadata format.
WORKER_MODEL_ENV=()
case "$HF_MODEL_ID" in
    *[Kk]imi*) WORKER_MODEL_ENV+=(-e BLAZE_SOCKET_DESCRIPTOR_PREFIX=kimi) ;;
    *)         WORKER_MODEL_ENV+=(-e USE_DEEPSEEK_MD_FORMAT=1 -e BLAZE_SOCKET_DESCRIPTOR_PREFIX=deepseek) ;;
esac

prefill_worker_name() {
    local idx="$1"
    if [[ "$idx" == "0" ]]; then
        printf '%s' "$PREFILL_WORKER_NAME"
    else
        printf '%s-%s' "$PREFILL_WORKER_NAME" "$idx"
    fi
}

start_prefill_worker() {
    local name="$1"
    local label="$2"
    local port="$3"
    shift 3

    local command=()
    if [[ -n "$LOCAL_BUILD" ]]; then
        command=(-c "cd cpp_server && ./build/tt_media_server_cpp -p ${port}")
    fi

    log "starting ${label} ($WORKER_IMAGE)"
    docker run -d --name "$name" --network "$NETWORK_NAME" --shm-size=2g \
        "${DEVICE_ARGS[@]}" "${LOCAL_BUILD_MOUNT[@]}" "${WORKER_ENTRYPOINT[@]}" \
        -e SERVER_MODE=cpp \
        -e MODEL="$HF_MODEL_ID" \
        -e LLM_DEVICE_BACKEND="$LLM_DEVICE_BACKEND" \
        -e LLM_MODE=prefill \
        -e DEVICE_IDS="$DEVICE_IDS" \
        "${WORKER_MODEL_ENV[@]}" \
        -e MAX_SESSIONS_COUNT=128 -e TT_LOG_LEVEL=debug -e USE_FAST_MODE=1 \
        "$@" \
        "$WORKER_IMAGE" \
        "${command[@]}" \
        >/dev/null
    PREFILL_WORKERS_STARTED+=("$name")

    sleep 2
    docker ps --format '{{.Names}}' | grep -q "^${name}\$" \
        || { docker logs --tail 80 "$name" >&2 || true; die "${label} exited during startup"; }
}

wait_dynamo_prefill_workers() {
    local expected="$1"
    local prefix="v1/instances/${DYNAMO_ROUTING_NAMESPACE:-dynamo}/prefill/generate/"

    log "waiting for ${expected} native prefill worker(s) to register with etcd (up to 60s)"
    for _ in $(seq 1 60); do
        local alive=1
        for name in "${PREFILL_WORKERS_STARTED[@]}"; do
            docker ps --format '{{.Names}}' | grep -q "^${name}\$" || alive=0
        done
        [[ "$alive" == "1" ]] || {
            for name in "${PREFILL_WORKERS_STARTED[@]}"; do docker logs --tail 80 "$name" >&2 || true; done
            die "native prefill worker exited before registering"
        }

        local registered
        registered="$(docker exec "$ETCD_NAME" etcdctl get --prefix --keys-only "$prefix" 2>/dev/null | grep -c "^${prefix}" || true)"
        if (( registered >= expected )); then
            log "native prefill worker(s) registered:"
            docker exec "$ETCD_NAME" etcdctl get --prefix --keys-only "v1/instances/${DYNAMO_ROUTING_NAMESPACE:-dynamo}/" | grep -v '^$' | sed 's/^/[deploy]   /'
            return
        fi
        sleep 1
    done

    for name in "${PREFILL_WORKERS_STARTED[@]}"; do docker logs --tail 80 "$name" >&2 || true; done
    die "native prefill worker(s) did not register within 60s"
}

if [[ "$PREFILL_GATEWAY_ENABLED" == "1" ]]; then
    PREFILL_CONNECT_PORT="$(endpoint_port "$PREFILL_GATEWAY_PREFILL_BIND")"

    for idx in $(seq 0 $((PREFILL_WORKER_COUNT - 1))); do
        start_prefill_worker "$(prefill_worker_name "$idx")" "managed gateway prefill worker ${idx}" "$PREFILL_WORKER_PORT" \
            -e DYNAMO_ENDPOINT_ENABLED=0 \
            -e USE_PREFILL_GATEWAY=1 \
            -e SOCKET_HOST="$PREFILL_GATEWAY_NAME" \
            -e SOCKET_PORT="$PREFILL_CONNECT_PORT" \
            -e PREFILL_SERVER_ID="managed-prefill-${idx}"
    done
fi

WORKER_ROUTING_ENV=()
if [[ "$PREFILL_GATEWAY_ENABLED" == "1" || "$PREFILL_DIRECT_ENABLED" == "1" || "$DYNAMO_ROUTING_ENABLED" == "1" ]]; then
    WORKER_ROUTING_ENV+=(
        -e LLM_MODE=decode
    )
fi
if [[ "$PREFILL_GATEWAY_ENABLED" == "1" ]]; then
    WORKER_ROUTING_ENV+=(
        -e USE_PREFILL_GATEWAY=1
        -e MAX_TOKENS_TO_PREFILL_ON_DECODE="${MAX_TOKENS_TO_PREFILL_ON_DECODE:-0}"
        -e SOCKET_HOST="$PREFILL_GATEWAY_NAME"
        -e SOCKET_PORT="$PREFILL_GATEWAY_DECODE_PORT"
    )
elif [[ "$PREFILL_DIRECT_ENABLED" == "1" ]]; then
    WORKER_ROUTING_ENV+=(
        -e USE_PREFILL_GATEWAY=0
        -e MAX_TOKENS_TO_PREFILL_ON_DECODE="${MAX_TOKENS_TO_PREFILL_ON_DECODE:-1000}"
        -e SOCKET_HOST=0.0.0.0
        -e SOCKET_PORT="$PREFILL_DIRECT_SOCKET_PORT"
    )
elif [[ "$DYNAMO_ROUTING_ENABLED" == "1" ]]; then
    WORKER_ROUTING_ENV+=(
        -e DYNAMO_NAMESPACE="${DYNAMO_ROUTING_NAMESPACE:-dynamo}"
        -e DYNAMO_COMPONENT=decode
        -e DYNAMO_ENDPOINT_NAME=generate
        -e DYNAMO_MODEL_TYPE=Chat
        -e DYNAMO_WORKER_TYPE=decode
        -e USE_PREFILL_GATEWAY=0
        -e DYNAMO_ROUTING=1
    )
fi

log "starting worker ($WORKER_IMAGE)"
docker run -d --name "$WORKER_NAME" --network "$NETWORK_NAME" --shm-size=2g \
    "${DEVICE_ARGS[@]}" "${LOCAL_BUILD_MOUNT[@]}" "${WORKER_ENTRYPOINT[@]}" \
    $(dynamo_worker_env "http://${ETCD_NAME}:2379" docker) \
    -e SERVER_MODE=cpp \
    -e MODEL="$HF_MODEL_ID" \
    -e LLM_DEVICE_BACKEND="$LLM_DEVICE_BACKEND" \
    -e DEVICE_IDS="$DEVICE_IDS" \
    "${WORKER_ROUTING_ENV[@]}" \
    "${WORKER_MODEL_ENV[@]}" \
    -e MAX_SESSIONS_COUNT=128 -e TT_LOG_LEVEL=debug -e USE_FAST_MODE=1 \
    -e MIN_TOKENS_TO_COPY="${MIN_TOKENS_TO_COPY:-1024}" \
    -e MOCK_PREFILL_SLEEP_MS="${MOCK_PREFILL_SLEEP_MS:-0}" \
    -e DYN_TX_TRACE="${DYN_TX_TRACE:-}" \
    "$WORKER_IMAGE" \
    ${LOCAL_BUILD:+-c 'cd cpp_server && LIB="$(pwd)/tt-llm-engine/build-full/libtt_llm_engine.so.0"; [ -f "$LIB" ] && export LD_PRELOAD="$LIB"; ./build/tt_media_server_cpp'} \
    >/dev/null

log "waiting for worker to register with etcd (up to 60s)"
for _ in $(seq 1 60); do
    docker exec "$ETCD_NAME" etcdctl get --prefix --keys-only v1/instances/ 2>/dev/null \
        | grep -q '^v1/instances/' && { REGISTERED=1; break; }
    docker ps --format '{{.Names}}' | grep -q "^${WORKER_NAME}\$" \
        || { docker logs --tail 80 "$WORKER_NAME" >&2 || true; die "worker exited before registering"; }
    sleep 1
done
[[ -n "${REGISTERED:-}" ]] || { docker logs --tail 80 "$WORKER_NAME" >&2 || true; die "worker did not register within 60s"; }
log "worker registered:"
docker exec "$ETCD_NAME" etcdctl get --prefix --keys-only v1/ | grep -v '^$' | sed 's/^/[deploy]   /'

if [[ "$PREFILL_DIRECT_ENABLED" == "1" ]]; then
    start_prefill_worker "$PREFILL_WORKER_NAME" "direct prefill worker" "$PREFILL_WORKER_PORT" \
        -e DYNAMO_ENDPOINT_ENABLED=0 \
        -e USE_PREFILL_GATEWAY=0 \
        -e SOCKET_HOST="$WORKER_NAME" \
        -e SOCKET_PORT="$PREFILL_DIRECT_SOCKET_PORT" \
        -e TT_MEMORY_REQUEST_QUEUE=tt_mem_requests_prefill \
        -e TT_MEMORY_RESULT_QUEUE=tt_mem_results_prefill \
        -e TT_WORKER_METRICS_SHM=/tt_worker_metrics_prefill
fi

if [[ "$DYNAMO_ROUTING_ENABLED" == "1" ]]; then
    for idx in $(seq 0 $((PREFILL_WORKER_COUNT - 1))); do
        start_prefill_worker "$(prefill_worker_name "$idx")" "Dynamo-registered prefill worker ${idx}" "$PREFILL_WORKER_PORT" \
            -e DYNAMO_ENDPOINT_ENABLED=1 \
            -e DYNAMO_ROUTING=1 \
            -e USE_PREFILL_GATEWAY=0 \
            -e DYNAMO_DISCOVERY_BACKEND=etcd \
            -e DYNAMO_ETCD_ENDPOINTS="http://${ETCD_NAME}:2379" \
            -e DYNAMO_NAMESPACE="${DYNAMO_ROUTING_NAMESPACE:-dynamo}" \
            -e DYNAMO_COMPONENT=prefill \
            -e DYNAMO_ENDPOINT_NAME=generate \
            -e DYNAMO_MODEL_TYPE=Prefill \
            -e DYNAMO_MODEL_INPUT=Tokens \
            -e DYNAMO_WORKER_TYPE=prefill \
            -e TT_MEMORY_REQUEST_QUEUE="tt_mem_requests_prefill_${idx}" \
            -e TT_MEMORY_RESULT_QUEUE="tt_mem_results_prefill_${idx}" \
            -e TT_WORKER_METRICS_SHM="/tt_worker_metrics_prefill_${idx}"
    done
    wait_dynamo_prefill_workers "$PREFILL_WORKER_COUNT"
fi

# ── frontend ────────────────────────────────────────────────────────────────
# The frontend image bakes the tokenizer tree at the path each worker advertises
# in its MDC, so the run is model-agnostic: it resolves every discovered model's
# tokenizer locally and serves whatever workers register in etcd.
FRONTEND_EXTRA_ENV=()
for env_name in \
    DYN_ROUTER_CONDITIONAL_DISAGG \
    DYN_ROUTER_CONDITIONAL_DISAGG_POLICY \
    DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD \
    DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD \
    DYN_ROUTER_CONDITIONAL_DISAGG_PREFILL_BUSY_THRESHOLD \
    DYN_ROUTER_CONDITIONAL_DISAGG_DECODE_BUSY_THRESHOLD; do
    if [[ -n "${!env_name:-}" ]]; then
        FRONTEND_EXTRA_ENV+=(-e "${env_name}=${!env_name}")
    fi
done

log "starting frontend ($FRONTEND_IMAGE)"
docker run -d --name "$FRONTEND_NAME" --network "$NETWORK_NAME" -p "${FRONTEND_HOST_PORT}:8000" \
    -e DYN_DISCOVERY_BACKEND=etcd \
    -e ETCD_ENDPOINTS="http://${ETCD_NAME}:2379" \
    -e DYN_REQUEST_PLANE_CODEC="${DYN_REQUEST_PLANE_CODEC:-json}" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e DYN_CHAT_PROCESSOR="${DYN_CHAT_PROCESSOR:-dynamo}" \
    -e DYN_RUNTIME_NUM_WORKER_THREADS="${DYN_RUNTIME_NUM_WORKER_THREADS:-}" \
    -e DYN_RUNTIME_MAX_BLOCKING_THREADS="${DYN_RUNTIME_MAX_BLOCKING_THREADS:-}" \
    -e DYN_COMPUTE_THREADS="${DYN_COMPUTE_THREADS:-}" \
    -e RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-}" \
    -e DYN_TOKENIZER="${DYN_TOKENIZER:-fastokens}" \
    -e DYN_DEBUG_PERF="${DYN_DEBUG_PERF:-0}" \
    -e DYN_ENABLE_ANTHROPIC_API="${DYN_ENABLE_ANTHROPIC_API:-true}" \
    -e ROUTER_MODE="${ROUTER_MODE:-kv}" \
    "${FRONTEND_EXTRA_ENV[@]}" \
    -e RUST_LOG="${RUST_LOG:-}" \
    "$FRONTEND_IMAGE" >/dev/null

if [[ "$MONITORING_ENABLED" == "1" ]]; then
    MONITORING_SERVER_TARGET="${SERVER_TARGET:-${FRONTEND_NAME}:8000}"
    MONITORING_SERVER_SERVICE="${SERVER_SERVICE:-dynamo_frontend}"
    if [[ "$PREFILL_GATEWAY_ENABLED" == "1" ]]; then
        MONITORING_GATEWAY_TARGET="${GATEWAY_TARGET:-${PREFILL_GATEWAY_NAME}:${PREFILL_GATEWAY_METRICS_PORT}}"
    else
        MONITORING_GATEWAY_TARGET="${GATEWAY_TARGET:-prefill-gateway:9091}"
    fi

    log "starting Prometheus + Grafana (scraping ${MONITORING_SERVER_TARGET})"
    TT_NET="$NETWORK_NAME" \
        PROMETHEUS_CONTAINER_NAME="$MONITORING_PROMETHEUS_NAME" \
        GRAFANA_CONTAINER_NAME="$MONITORING_GRAFANA_NAME" \
        PROCESS_EXPORTER_CONTAINER_NAME="$MONITORING_PROCESS_EXPORTER_NAME" \
        PROMETHEUS_HOST_PORT="$PROMETHEUS_HOST_PORT" \
        GRAFANA_HOST_PORT="$GRAFANA_HOST_PORT" \
        SERVER_TARGET="$MONITORING_SERVER_TARGET" \
        SERVER_SERVICE="$MONITORING_SERVER_SERVICE" \
        GATEWAY_TARGET="$MONITORING_GATEWAY_TARGET" \
        GF_HOME_DASHBOARD="${GF_HOME_DASHBOARD:-}" \
        docker compose -p "$MONITORING_PROJECT_NAME" -f "$MONITORING_COMPOSE" up -d >/dev/null
    MONITORING_STARTED=1
fi

log "frontend on http://localhost:${FRONTEND_HOST_PORT}"
if [[ "$MONITORING_ENABLED" == "1" ]]; then
    log "Prometheus on http://localhost:${PROMETHEUS_HOST_PORT}; Grafana on http://localhost:${GRAFANA_HOST_PORT} (admin/admin)"
fi
if [[ "$PREFILL_GATEWAY_ENABLED" == "1" ]]; then
    log "PrefillGateway metrics on ${PREFILL_GATEWAY_NAME}:${PREFILL_GATEWAY_METRICS_PORT} inside ${NETWORK_NAME}"
fi
if [[ "$DYNAMO_ROUTING_ENABLED" == "1" ]]; then
    log "Dynamo routing enabled; Dynamo owns local-vs-remote prefill routing"
fi
log "tailing worker logs (Ctrl+C to tear down)"
docker logs -f "$WORKER_NAME"
