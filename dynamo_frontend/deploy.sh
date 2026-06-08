#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fixed names/ports (not CLI-configurable).
NETWORK_NAME="dynamo-net"
ETCD_NAME="etcd"
WORKER_NAME="tt-cpp-worker"
FRONTEND_NAME="dynamo-frontend"
FRONTEND_HOST_PORT="8080"
MODEL_NAME="tt-cpp-server"
LLM_DEVICE_BACKEND="pipeline_manager"
WORKER_TOKENIZER_DIR="/home/container_app_user/app/server/cpp_server/tokenizers"

# Image defaults (override with the matching flag if needed).
ETCD_IMAGE="quay.io/coreos/etcd:v3.5.13"
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

log() { printf '[deploy] %s\n' "$*"; }
die() { printf '[deploy] %s\n' "$*" >&2; exit 1; }

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

HF_TOKEN and perf knobs (DYN_TOKENIZER, RAYON_NUM_THREADS, DYN_RUNTIME_*,
RUST_LOG, DYN_TX_TRACE, DYN_ENABLE_ANTHROPIC_API) are read from the environment.
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
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

# Local-build: validate the binary and prepare the bind-mount + entrypoint.
LOCAL_BUILD_MOUNT=()
WORKER_ENTRYPOINT=()
if [[ -n "$LOCAL_BUILD" ]]; then
    CPP_SERVER_DIR_ABS="$(readlink -f "$CPP_SERVER_DIR" 2>/dev/null || true)"
    [[ -d "$CPP_SERVER_DIR_ABS" ]] || die "cpp_server directory not found: $CPP_SERVER_DIR"
    [[ -f "$CPP_SERVER_DIR_ABS/build/tt_media_server_cpp" ]] \
        || die "no binary at $CPP_SERVER_DIR_ABS/build/tt_media_server_cpp — run ./build.sh first"
    log "using local build from $CPP_SERVER_DIR_ABS"
    LOCAL_BUILD_MOUNT+=(-v "${CPP_SERVER_DIR_ABS}/build:/home/container_app_user/app/server/cpp_server/build:ro")
    WORKER_ENTRYPOINT+=(--entrypoint /bin/bash)
fi

cleanup() {
    log "tearing down"
    docker rm -f "$FRONTEND_NAME" "$WORKER_NAME" "$ETCD_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker network inspect "$NETWORK_NAME" >/dev/null 2>&1 \
    || { log "creating network $NETWORK_NAME"; docker network create "$NETWORK_NAME" >/dev/null; }

# ── etcd ──────────────────────────────────────────────────────────────────
log "starting etcd ($ETCD_IMAGE)"
docker run -d --name "$ETCD_NAME" --network "$NETWORK_NAME" -p 2379:2379 \
    "$ETCD_IMAGE" /usr/local/bin/etcd --name dyn-etcd \
        --advertise-client-urls http://0.0.0.0:2379 \
        --listen-client-urls http://0.0.0.0:2379 >/dev/null

log "waiting for etcd"
for _ in $(seq 1 30); do
    docker exec "$ETCD_NAME" etcdctl endpoint health >/dev/null 2>&1 && { ETCD_OK=1; break; }
    sleep 1
done
[[ -n "${ETCD_OK:-}" ]] || { docker logs --tail 50 "$ETCD_NAME" >&2 || true; die "etcd never became healthy"; }
log "etcd healthy"

# ── worker ────────────────────────────────────────────────────────────────
DEVICE_ARGS=()
[[ -e /dev/tenstorrent ]] && DEVICE_ARGS+=(--device /dev/tenstorrent --cap-add=SYS_NICE)

# Model-specific env: Kimi uses the 'kimi' blaze prefix + default MD format;
# DeepSeek uses the legacy MD format + 'deepseek' prefix.
WORKER_MODEL_ENV=()
case "$HF_MODEL_ID" in
    *[Kk]imi*) WORKER_MODEL_ENV+=(-e BLAZE_SOCKET_DESCRIPTOR_PREFIX=kimi) ;;
    *)         WORKER_MODEL_ENV+=(-e USE_DEEPSEEK_MD_FORMAT=1 -e BLAZE_SOCKET_DESCRIPTOR_PREFIX=deepseek) ;;
esac

log "starting worker ($WORKER_IMAGE)"
docker run -d --name "$WORKER_NAME" --network "$NETWORK_NAME" --shm-size=2g \
    "${DEVICE_ARGS[@]}" "${LOCAL_BUILD_MOUNT[@]}" "${WORKER_ENTRYPOINT[@]}" \
    -e DYNAMO_ENDPOINT_ENABLED=1 \
    -e DYNAMO_ENABLE_RESPONSES_API=true \
    -e DYNAMO_DISCOVERY_BACKEND=etcd \
    -e DYNAMO_ETCD_ENDPOINTS="http://${ETCD_NAME}:2379" \
    -e DYNAMO_NAMESPACE=default -e DYNAMO_COMPONENT=backend -e DYNAMO_ENDPOINT_NAME=generate \
    -e SERVER_MODE=cpp \
    -e MODEL="$HF_MODEL_ID" \
    -e LLM_DEVICE_BACKEND="$LLM_DEVICE_BACKEND" \
    -e DEVICE_IDS="$DEVICE_IDS" \
    "${WORKER_MODEL_ENV[@]}" \
    -e MAX_SESSIONS_COUNT=128 -e TT_LOG_LEVEL=debug -e USE_FAST_MODE=1 \
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

# ── frontend ────────────────────────────────────────────────────────────────
# The frontend image bakes the tokenizer tree at WORKER_TOKENIZER_DIR (same
# fetch_tokenizers.sh the worker uses), so MODEL_PATH just points the entrypoint
# at the baked dir — no tokenizer extraction or bind-mount needed.
log "starting frontend ($FRONTEND_IMAGE)"
docker run -d --name "$FRONTEND_NAME" --network "$NETWORK_NAME" -p "${FRONTEND_HOST_PORT}:8000" \
    -e MODEL_PATH="${WORKER_TOKENIZER_DIR}/${HF_MODEL_ID}" \
    -e DYN_DISCOVERY_BACKEND=etcd \
    -e ETCD_ENDPOINTS="http://${ETCD_NAME}:2379" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e HF_MODEL_ID="$HF_MODEL_ID" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e DYN_CHAT_PROCESSOR="${DYN_CHAT_PROCESSOR:-dynamo}" \
    -e DYN_RUNTIME_NUM_WORKER_THREADS="${DYN_RUNTIME_NUM_WORKER_THREADS:-}" \
    -e DYN_RUNTIME_MAX_BLOCKING_THREADS="${DYN_RUNTIME_MAX_BLOCKING_THREADS:-}" \
    -e DYN_COMPUTE_THREADS="${DYN_COMPUTE_THREADS:-}" \
    -e RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-}" \
    -e DYN_TOKENIZER="${DYN_TOKENIZER:-fastokens}" \
    -e DYN_DEBUG_PERF="${DYN_DEBUG_PERF:-0}" \
    -e DYN_ENABLE_ANTHROPIC_API="${DYN_ENABLE_ANTHROPIC_API:-true}" \
    -e RUST_LOG="${RUST_LOG:-}" \
    "$FRONTEND_IMAGE" >/dev/null

log "frontend on http://localhost:${FRONTEND_HOST_PORT} — tailing worker logs (Ctrl+C to tear down)"
docker logs -f "$WORKER_NAME"
