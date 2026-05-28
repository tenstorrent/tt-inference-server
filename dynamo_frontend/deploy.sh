#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 --etcd-image <img> --worker-image <img> --frontend-image <img> [options]

Required:
  --etcd-image <img>          etcd Docker image (e.g. quay.io/coreos/etcd:v3.5.13)
  --worker-image <img>        cpp_server Docker image (e.g. tt-media-server-cpp:blaze)
  --frontend-image <img>      Dynamo frontend Docker image (e.g. dynamo-frontend)

Optional:
  --network-name <name>       (default: dynamo-net)
  --etcd-name <name>          (default: etcd)
  --worker-name <name>        (default: tt-cpp-worker)
  --frontend-name <name>      (default: dynamo-frontend)
  --frontend-host-port <port> (default: 8080)
  --worker-host-port <port>   host port to publish the worker's drogon HTTP
                              listener (container port 8000) on. Useful for
                              hitting cpp_server directly with curl / vllm
                              bench serve. (default: 8000; override when host
                              port 8000 is in use). The blaze socket
                              (container port 9000) is always published 1:1.
  --model-name <name>         (default: tt-cpp-server)
  --hf-model-id <id>          (default: deepseek-ai/DeepSeek-R1-0528)
  --llm-device-backend <name> (default: pipeline_manager)
  --device-ids <ids>          cpp_server DEVICE_IDS env. Two formats:
                              "(0),(1),(2),(3)" runs 4 LLMService consumer
                              threads with 1 device each; "10,11,14,15,…"
                              runs 1 consumer with N visible devices. The
                              TT-hardware launches use the comma-only form.
                              (default: "10,11,14,15,18,19,22,23" — the
                              local 8-device DeepSeek-R1 mesh)
  --tokenizers-host-dir <dir> host directory containing the tokenizers tree
                              (must hold <hf-model-id>/config.json,
                              tokenizer_config.json, and either tokenizer.json
                              or tiktoken.model for tiktoken-based models such
                              as Kimi K2.6). Bind-mounted into the
                              frontend at the path the worker advertises in
                              the MDC. (default: auto — extracts tokenizer
                              assets out of the worker image into a host
                              tmpdir and fetches a missing config.json
                              from HuggingFace, so no local checkout is
                              needed)
  --skip-tokenizer-share      don't bind-mount tokenizers into the frontend
                              (debug only — discovery-driven model loads will
                              fail with a HuggingFace 404)
  -h, --help                  show this help and exit

The HF_TOKEN environment variable, if set in the calling shell, is
forwarded to the frontend container for gated HuggingFace models.

Performance knobs (read from the calling shell, optional):
  DYN_TOKENIZER         BPE tokenizer backend: 'default' (HuggingFace
                        tokenizers crate, single-flight per request and
                        the suspected serializer on concurrent benches) or
                        'fastokens' (in-house BPE crate, parallel-safe).
                        (default: fastokens — flip back to 'default' to
                        confirm whether your test regression is the
                        tokenizer.)
  RAYON_NUM_THREADS     size of the global Rayon pool that HF's
                        tokenizers crate uses internally. Default in the
                        container: nproc.
  DYN_RUNTIME_NUM_WORKER_THREADS,
  DYN_RUNTIME_MAX_BLOCKING_THREADS,
  DYN_COMPUTE_THREADS   Tokio I/O / blocking / compute pool sizes.
  RUST_LOG              tracing filter for the frontend, e.g.
                          info
                          info,dynamo_llm::preprocessor=debug,dynamo_runtime::transport=debug
                        to see per-request spans and find where the
                        ~600 ms TTFT delta is hiding.
  DYN_TX_TRACE          Enable per-token send tracing on the worker
                        (cpp_server) side. When set to '1' (or any
                        non-zero/non-'false' value) the worker emits a
                        '[DynamoTx] stage=chunk seq=N ... since_prev_us=...'
                        log line for every TokenChunk it writes to the
                        frontend's call-home socket. Use to verify
                        whether inter-token latency lives on the
                        backend->frontend wire or downstream of it. High
                        log volume (~one line per generated token), keep
                        off in normal runs.
  DYN_ENABLE_ANTHROPIC_API
                        Enable experimental Anthropic Messages API endpoint
                        (/v1/messages). (default: true)

Example:
  $0 \\
    --etcd-image quay.io/coreos/etcd:v3.5.13 \\
    --worker-image tt-media-server-cpp:blaze \\
    --frontend-image dynamo-frontend
EOF
    exit 1
}

ETCD_IMAGE=""
WORKER_IMAGE=""
FRONTEND_IMAGE=""
NETWORK_NAME="dynamo-net"
ETCD_NAME="etcd"
WORKER_NAME="tt-cpp-worker"
FRONTEND_NAME="dynamo-frontend"
FRONTEND_HOST_PORT="8080"
WORKER_HOST_PORT="8000"
MODEL_NAME="tt-cpp-server"
HF_MODEL_ID="deepseek-ai/DeepSeek-R1-0528"
LLM_DEVICE_BACKEND="pipeline_manager"
DEVICE_IDS="10,11,14,15,18,19,22,23"
TOKENIZERS_HOST_DIR=""
SKIP_TOKENIZER_SHARE=""
WORKER_TOKENIZER_DIR="/home/container_app_user/app/server/cpp_server/tokenizers"
TOKENIZERS_TEMP_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --etcd-image)         ETCD_IMAGE="$2";         shift 2 ;;
        --worker-image)       WORKER_IMAGE="$2";       shift 2 ;;
        --frontend-image)     FRONTEND_IMAGE="$2";     shift 2 ;;
        --network-name)       NETWORK_NAME="$2";       shift 2 ;;
        --etcd-name)          ETCD_NAME="$2";          shift 2 ;;
        --worker-name)        WORKER_NAME="$2";        shift 2 ;;
        --frontend-name)      FRONTEND_NAME="$2";      shift 2 ;;
        --frontend-host-port) FRONTEND_HOST_PORT="$2"; shift 2 ;;
        --worker-host-port)   WORKER_HOST_PORT="$2";   shift 2 ;;
        --model-name)         MODEL_NAME="$2";         shift 2 ;;
        --hf-model-id)        HF_MODEL_ID="$2";        shift 2 ;;
        --llm-device-backend) LLM_DEVICE_BACKEND="$2"; shift 2 ;;
        --device-ids)         DEVICE_IDS="$2";         shift 2 ;;
        --tokenizers-host-dir) TOKENIZERS_HOST_DIR="$2"; shift 2 ;;
        --skip-tokenizer-share) SKIP_TOKENIZER_SHARE=1; shift ;;
        -h|--help)            usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "$ETCD_IMAGE" || -z "$WORKER_IMAGE" || -z "$FRONTEND_IMAGE" ]]; then
    echo "Missing required argument(s)." >&2
    usage
fi

log() { printf '[deploy] %s\n' "$*"; }

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

log "publishing worker HTTP: host:${WORKER_HOST_PORT} -> container:8000"
log "publishing worker blaze socket: host:9000 -> container:9000"

log "starting worker ($WORKER_IMAGE)"
# --privileged + the /dev/hugepages + /var/run/tenstorrent + /lib/modules
# mounts give cpp_server access to the TT devices. --ipc=host shares the
# host's /dev/shm so Boost.Interprocess queues between the supervisor and
# the fork+exec'd worker processes are not capped by the 64 MiB container
# default. The ulimits keep the supervisor from running out of fds/threads
# at high session counts. These flags mirror the standalone cpp_server
# launch script.
docker run -d --name "$WORKER_NAME" \
    --network "$NETWORK_NAME" \
    --privileged \
    --ipc=host \
    --ulimit nofile=65536:65536 \
    --ulimit nproc=65536:65536 \
    -p "${WORKER_HOST_PORT}:8000" \
    -p 9000:9000 \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v /etc/udev/rules.d:/etc/udev/rules.d \
    -v /lib/modules:/lib/modules \
    -v /var/run/tenstorrent:/var/run/tenstorrent \
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
    -e USE_DEEPSEEK_MD_FORMAT=1 \
    -e BLAZE_SOCKET_DESCRIPTOR_PREFIX=deepseek \
    -e MAX_SESSIONS_COUNT=128 \
    -e TT_LOG_LEVEL=debug \
    -e USE_FAST_MODE=1 \
    -e DYN_TX_TRACE="${DYN_TX_TRACE:-}" \
    --entrypoint /bin/bash \
    "$WORKER_IMAGE" \
    -c 'cd cpp_server && LD_PRELOAD=$(pwd)/tt-llm-engine/build-full/libtt_llm_engine.so.0 ./build/tt_media_server_cpp' \
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
# frontend, so we collect them onto the host and bind-mount that dir into
# the frontend.
#
# Sources, in priority order:
#   1. --tokenizers-host-dir <path>   user-provided host tree
#   2. auto-extract from $WORKER_IMAGE (default) — pulls whatever
#      cpp_server/build.sh's pre-fetch put under $WORKER_TOKENIZER_DIR
#      into a tmpdir. The image ships tokenizer.json + tokenizer_config.json
#      but NOT the HF model config.json, so we backfill that from HF (or
#      synthesize a minimal placeholder if HF is unreachable). Pointing
#      $MODEL_PATH at the same dir lets the frontend entrypoint skip its
#      own HF download.
TOKENIZER_MOUNT=()
FRONTEND_MODEL_PATH_ENV=()
if [[ -z "$SKIP_TOKENIZER_SHARE" ]]; then
    if [[ -n "$TOKENIZERS_HOST_DIR" ]]; then
        TOKENIZERS_HOST_DIR_ABS="$(readlink -f "$TOKENIZERS_HOST_DIR" 2>/dev/null || true)"
        if [[ -z "$TOKENIZERS_HOST_DIR_ABS" || ! -d "$TOKENIZERS_HOST_DIR_ABS" ]]; then
            log "tokenizers host dir not found: $TOKENIZERS_HOST_DIR"
            log "  drop --tokenizers-host-dir to auto-extract from worker image"
            exit 1
        fi
    else
        TOKENIZERS_TEMP_DIR="$(mktemp -d --suffix=-deploy-tokenizers)"
        log "extracting tokenizers from worker image -> $TOKENIZERS_TEMP_DIR"
        TMP_CID="$(docker create "$WORKER_IMAGE")"
        if ! docker cp "${TMP_CID}:${WORKER_TOKENIZER_DIR}/." "$TOKENIZERS_TEMP_DIR/" >/dev/null 2>&1; then
            docker rm "$TMP_CID" >/dev/null 2>&1 || true
            log "worker image has no tokenizers at ${WORKER_TOKENIZER_DIR}"
            log "  pass --tokenizers-host-dir <path> with the tokenizer tree"
            exit 1
        fi
        docker rm "$TMP_CID" >/dev/null 2>&1 || true
        TOKENIZERS_HOST_DIR_ABS="$TOKENIZERS_TEMP_DIR"
    fi

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
    TOKENIZER_MOUNT+=(-v "${TOKENIZERS_HOST_DIR_ABS}:${WORKER_TOKENIZER_DIR}:ro")
    FRONTEND_MODEL_PATH_ENV+=(-e "MODEL_PATH=${WORKER_TOKENIZER_DIR}/${HF_MODEL_ID}")
else
    log "skipping tokenizer share (--skip-tokenizer-share)"
fi

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
log "tailing frontend logs (Ctrl+C to tear down)"
docker logs -f "$FRONTEND_NAME"
