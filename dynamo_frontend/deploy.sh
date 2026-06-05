#!/usr/bin/env bash
set -euo pipefail

# Directory this script lives in. Used as the base for the tokenizer scratch
# dir: it must NOT be under /tmp, because on hosts where dockerd runs with
# systemd PrivateTmp a `-v /tmp/x:…` bind mount sees the daemon's (empty) /tmp
# rather than ours, silently producing an empty tokenizer mount (the frontend
# then 404s the model). See DEPLOY.md / MANUAL_KIMI_MOCK.md §D.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Fixed names / ports (no longer CLI-configurable) ───────────────────────
NETWORK_NAME="dynamo-net"
ETCD_NAME="etcd"
WORKER_NAME="tt-cpp-worker"
FRONTEND_NAME="dynamo-frontend"
FRONTEND_HOST_PORT="8080"
MODEL_NAME="tt-cpp-server"
LLM_DEVICE_BACKEND="pipeline_manager"
WORKER_TOKENIZER_DIR="/home/container_app_user/app/server/cpp_server/tokenizers"

# ── Image defaults (all optional; override only if you need to) ─────────────
ETCD_IMAGE="quay.io/coreos/etcd:v3.5.13"
WORKER_IMAGE="tt-cpp-worker:mock"
FRONTEND_IMAGE="dynamo-frontend:latest"

# ── Model selection ─────────────────────────────────────────────────────────
KIMI_MODEL_ID="moonshotai/Kimi-K2.6"
DEEPSEEK_MODEL_ID="deepseek-ai/DeepSeek-R1-0528"
HF_MODEL_ID="$DEEPSEEK_MODEL_ID"   # default

DEVICE_IDS="10,11,14,15,18,19,22,23"

# Local-build: bind-mount this repo's cpp_server/build over the worker image
# and run the local binary. Defaults to the in-repo cpp_server next to this
# script, so --local-build needs no extra path argument.
LOCAL_BUILD=""
CPP_SERVER_DIR="${SCRIPT_DIR}/../tt-media-server/cpp_server"

TOKENIZERS_TEMP_DIR=""

usage() {
    cat >&2 <<EOF
Usage: $0 [options]

Model (pick one; default: --deepseek):
  --kimi                      serve ${KIMI_MODEL_ID}
  --deepseek                  serve ${DEEPSEEK_MODEL_ID}
  --hf-model-id <id>          explicit HF model id (overrides --kimi/--deepseek)

Build:
  --local-build               bind-mount <repo>/tt-media-server/cpp_server/build
                              over the worker image and run the local binary
                              (no --worker-image needed)

Images (all optional; sensible defaults):
  --etcd-image <img>          (default: ${ETCD_IMAGE})
  --worker-image <img>        (default: ${WORKER_IMAGE})
  --frontend-image <img>      (default: ${FRONTEND_IMAGE})

Other:
  --device-ids <ids>          cpp_server DEVICE_IDS env (default: ${DEVICE_IDS})

The HF_TOKEN env var, if set in the calling shell, is forwarded to the
frontend for gated HuggingFace models. Performance knobs (DYN_TOKENIZER,
RAYON_NUM_THREADS, DYN_RUNTIME_*, RUST_LOG, DYN_TX_TRACE, DYN_ENABLE_ANTHROPIC_API)
are read from the environment if set.

Examples:
  $0 --deepseek
  $0 --kimi --local-build
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

log() { printf '[deploy] %s\n' "$*"; }

if [[ -n "$LOCAL_BUILD" ]]; then
    CPP_SERVER_DIR_ABS="$(readlink -f "$CPP_SERVER_DIR" 2>/dev/null || true)"
    if [[ -z "$CPP_SERVER_DIR_ABS" || ! -d "$CPP_SERVER_DIR_ABS" ]]; then
        echo "cpp_server directory not found: $CPP_SERVER_DIR" >&2
        exit 1
    fi
    if [[ ! -f "$CPP_SERVER_DIR_ABS/build/tt_media_server_cpp" ]]; then
        echo "cpp_server binary not found at $CPP_SERVER_DIR_ABS/build/tt_media_server_cpp" >&2
        echo "  run ./build.sh in the cpp_server directory first" >&2
        exit 1
    fi
    log "using local build from $CPP_SERVER_DIR_ABS"
fi

cleanup() {
    log "tearing down"
    docker rm -f "$FRONTEND_NAME" "$WORKER_NAME" "$ETCD_NAME" >/dev/null 2>&1 || true
    if [[ -n "$TOKENIZERS_TEMP_DIR" && -d "$TOKENIZERS_TEMP_DIR" ]]; then
        rm -rf "$TOKENIZERS_TEMP_DIR"
    fi
}
trap cleanup EXIT INT TERM

if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
    log "creating network $NETWORK_NAME"
    docker network create "$NETWORK_NAME" >/dev/null
else
    log "network $NETWORK_NAME already exists"
fi

log "starting etcd ($ETCD_IMAGE)"
docker run -d --name "$ETCD_NAME" \
    --network "$NETWORK_NAME" \
    -p 2379:2379 \
    "$ETCD_IMAGE" \
    /usr/local/bin/etcd \
        --name dyn-etcd \
        --advertise-client-urls http://0.0.0.0:2379 \
        --listen-client-urls http://0.0.0.0:2379 >/dev/null

log "waiting for etcd to become healthy"
for _ in $(seq 1 30); do
    if docker exec "$ETCD_NAME" etcdctl endpoint health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! docker exec "$ETCD_NAME" etcdctl endpoint health 2>&1 | tee /dev/stderr | grep -q 'is healthy'; then
    log "etcd never became healthy; recent etcd logs:"
    docker logs --tail 50 "$ETCD_NAME" >&2 || true
    exit 1
fi
log "etcd healthy"

log "starting worker ($WORKER_IMAGE)"

# Device args for TT hardware
DEVICE_ARGS=()
if [[ -e /dev/tenstorrent ]]; then
    DEVICE_ARGS+=(--device /dev/tenstorrent --cap-add=SYS_NICE)
fi

# Build mount args for local build - only mount the build directory to preserve
# container's tt-llm-engine and other dependencies
LOCAL_BUILD_MOUNT=()
WORKER_ENTRYPOINT=()
if [[ -n "$LOCAL_BUILD" ]]; then
    log "bind-mounting local build: $CPP_SERVER_DIR_ABS/build -> container cpp_server/build"
    LOCAL_BUILD_MOUNT+=(-v "${CPP_SERVER_DIR_ABS}/build:/home/container_app_user/app/server/cpp_server/build:ro")
    WORKER_ENTRYPOINT+=(--entrypoint /bin/bash)
fi

# Model-specific worker env. DeepSeek-R1 uses the legacy MD format and the
# 'deepseek' blaze socket descriptor prefix. Kimi K2.6 uses its own 'kimi'
# prefix and the default MD format (its tiktoken tokenizer is advertised as
# tik_token_model, not DeepSeek's hf_tokenizer_json layout).
WORKER_MODEL_ENV=()
case "$HF_MODEL_ID" in
    *[Kk]imi*)
        WORKER_MODEL_ENV+=(-e BLAZE_SOCKET_DESCRIPTOR_PREFIX=kimi)
        ;;
    *)
        WORKER_MODEL_ENV+=(-e USE_DEEPSEEK_MD_FORMAT=1)
        WORKER_MODEL_ENV+=(-e BLAZE_SOCKET_DESCRIPTOR_PREFIX=deepseek)
        ;;
esac

docker run -d --name "$WORKER_NAME" \
    --network "$NETWORK_NAME" \
    --shm-size=2g \
    "${DEVICE_ARGS[@]}" \
    "${LOCAL_BUILD_MOUNT[@]}" \
    "${WORKER_ENTRYPOINT[@]}" \
    -e DYNAMO_ENDPOINT_ENABLED=1 \
    -e DYNAMO_DISCOVERY_BACKEND=etcd \
    -e DYNAMO_ETCD_ENDPOINTS="http://${ETCD_NAME}:2379" \
    -e DYNAMO_NAMESPACE=default \
    -e DYNAMO_COMPONENT=backend \
    -e DYNAMO_ENDPOINT_NAME=generate \
    -e SERVER_MODE=cpp \
    -e MODEL="$HF_MODEL_ID" \
    -e LLM_DEVICE_BACKEND="$LLM_DEVICE_BACKEND" \
    -e DEVICE_IDS="$DEVICE_IDS" \
    "${WORKER_MODEL_ENV[@]}" \
    -e MAX_SESSIONS_COUNT=128 \
    -e TT_LOG_LEVEL=debug \
    -e USE_FAST_MODE=1 \
    -e DYN_TX_TRACE="${DYN_TX_TRACE:-}" \
    "$WORKER_IMAGE" \
    ${LOCAL_BUILD:+-c 'cd cpp_server && LIB="$(pwd)/tt-llm-engine/build-full/libtt_llm_engine.so.0"; [ -f "$LIB" ] && export LD_PRELOAD="$LIB"; ./build/tt_media_server_cpp'} \
    >/dev/null

log "waiting for worker to register against etcd (up to 60s)"
REGISTERED=""
for _ in $(seq 1 60); do
    if docker exec "$ETCD_NAME" etcdctl get --prefix --keys-only v1/instances/ 2>/dev/null \
       | grep -q '^v1/instances/'; then
        REGISTERED=1
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${WORKER_NAME}\$"; then
        log "worker container exited before registration; recent logs:"
        docker logs --tail 80 "$WORKER_NAME" >&2 || true
        exit 1
    fi
    sleep 1
done

if [[ -z "$REGISTERED" ]]; then
    log "worker did not register within 60s; recent worker logs:"
    docker logs --tail 80 "$WORKER_NAME" >&2 || true
    exit 1
fi

log "worker registered the following keys:"
docker exec "$ETCD_NAME" etcdctl get --prefix --keys-only v1/ \
    | grep -v '^$' | sed 's/^/[deploy]   /'

# The worker advertises absolute paths (inside *its* container) for the model
# config / tokenizer files in its MDC. Those paths don't exist in the frontend
# container, and Dynamo's loader then treats them as HF repo ids and fails
# with a 404. We need the same files at the same absolute path inside the
# frontend, so we extract them from the worker image onto the host and
# bind-mount that dir into the frontend. The image ships tokenizer.json +
# tokenizer_config.json (and config.json); pointing $MODEL_PATH at the same
# dir lets the frontend entrypoint skip its own HF download.
TOKENIZER_MOUNT=()
FRONTEND_MODEL_PATH_ENV=()

# NOT under /tmp — see SCRIPT_DIR note above (dockerd PrivateTmp would make the
# resulting bind mount silently empty).
TOKENIZERS_TEMP_DIR="$(mktemp -d "${SCRIPT_DIR}/.deploy-tokenizers.XXXXXX")"
log "extracting tokenizers from worker image -> $TOKENIZERS_TEMP_DIR"
TMP_CID="$(docker create "$WORKER_IMAGE")"
if ! docker cp "${TMP_CID}:${WORKER_TOKENIZER_DIR}/." "$TOKENIZERS_TEMP_DIR/" >/dev/null 2>&1; then
    docker rm "$TMP_CID" >/dev/null 2>&1 || true
    log "worker image has no tokenizers at ${WORKER_TOKENIZER_DIR}"
    exit 1
fi
docker rm "$TMP_CID" >/dev/null 2>&1 || true
TOKENIZERS_HOST_DIR_ABS="$TOKENIZERS_TEMP_DIR"

MODEL_SUBDIR="$TOKENIZERS_HOST_DIR_ABS/$HF_MODEL_ID"
mkdir -p "$MODEL_SUBDIR"

# Validate required tokenizer files exist (expected to be in worker image)
for f in config.json tokenizer_config.json; do
    if [[ ! -f "$MODEL_SUBDIR/$f" ]]; then
        log "missing $f under $MODEL_SUBDIR"
        log "  ensure tokenizer files are included in the worker image"
        exit 1
    fi
done
if [[ ! -f "$MODEL_SUBDIR/tokenizer.json" && ! -f "$MODEL_SUBDIR/tiktoken.model" ]]; then
    log "missing tokenizer.json or tiktoken.model under $MODEL_SUBDIR"
    log "  ensure tokenizer files are included in the worker image"
    exit 1
fi
if [[ "$HF_MODEL_ID" == *Kimi* || "$HF_MODEL_ID" == *kimi* ]]; then
    if [[ ! -f "$MODEL_SUBDIR/chat_template.jinja" ]]; then
        log "missing chat_template.jinja under $MODEL_SUBDIR"
        log "  ensure tokenizer files are included in the worker image"
        exit 1
    fi
fi

log "mounting tokenizers: $TOKENIZERS_HOST_DIR_ABS -> $WORKER_TOKENIZER_DIR"
log "  using $HF_MODEL_ID:"
find "$MODEL_SUBDIR" -maxdepth 1 -type f \
    | sed "s|^|[deploy]     |"
# Mount read-write (not :ro): the frontend entrypoint.sh runs
# `mkdir -p "$MODEL_PATH"` under `set -e`, and on a read-only bind mount that
# mkdir fails with EROFS (the kernel reports the error even though the dir
# already exists), so the frontend container exits before starting. RW lets
# the no-op mkdir succeed; nothing is actually written for the baked models.
TOKENIZER_MOUNT+=(-v "${TOKENIZERS_HOST_DIR_ABS}:${WORKER_TOKENIZER_DIR}:rw")
FRONTEND_MODEL_PATH_ENV+=(-e "MODEL_PATH=${WORKER_TOKENIZER_DIR}/${HF_MODEL_ID}")

log "starting frontend ($FRONTEND_IMAGE)"
docker run -d --name "$FRONTEND_NAME" \
    --network "$NETWORK_NAME" \
    -p "${FRONTEND_HOST_PORT}:8000" \
    "${TOKENIZER_MOUNT[@]}" \
    "${FRONTEND_MODEL_PATH_ENV[@]}" \
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

log "frontend reachable on http://localhost:${FRONTEND_HOST_PORT}"
log "tailing worker logs (Ctrl+C to tear down)"
docker logs -f "$WORKER_NAME"
