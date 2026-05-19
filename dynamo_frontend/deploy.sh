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
  --model-name <name>         (default: tt-cpp-server)
  --hf-model-id <id>          (default: meta-llama/Llama-3.1-8B-Instruct)
  --llm-device-backend <name> (default: mock_pipeline)
  -h, --help                  show this help and exit

The HF_TOKEN environment variable, if set in the calling shell, is
forwarded to the frontend container for gated HuggingFace models.

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
MODEL_NAME="tt-cpp-server"
HF_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
LLM_DEVICE_BACKEND="mock_pipeline"

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
        --model-name)         MODEL_NAME="$2";         shift 2 ;;
        --hf-model-id)        HF_MODEL_ID="$2";        shift 2 ;;
        --llm-device-backend) LLM_DEVICE_BACKEND="$2"; shift 2 ;;
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

DEVICE_ARGS=()
if [[ -e /dev/tenstorrent ]]; then
    DEVICE_ARGS+=(--device /dev/tenstorrent --cap-add=SYS_NICE)
fi

log "starting worker ($WORKER_IMAGE)"
docker run -d --name "$WORKER_NAME" \
    --network "$NETWORK_NAME" \
    --shm-size=2g \
    "${DEVICE_ARGS[@]}" \
    -e DYNAMO_ENDPOINT_ENABLED=1 \
    -e DYNAMO_DISCOVERY_BACKEND=etcd \
    -e DYNAMO_ETCD_ENDPOINTS="http://${ETCD_NAME}:2379" \
    -e DYNAMO_NAMESPACE=default \
    -e DYNAMO_COMPONENT=backend \
    -e DYNAMO_ENDPOINT_NAME=generate \
    -e DYN_TCP_RPC_HOST="$WORKER_NAME" \
    -e SERVER_MODE=cpp \
    -e LLM_DEVICE_BACKEND="$LLM_DEVICE_BACKEND" \
    "$WORKER_IMAGE" >/dev/null

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

log "starting frontend ($FRONTEND_IMAGE)"
docker run -d --name "$FRONTEND_NAME" \
    --network "$NETWORK_NAME" \
    -p "${FRONTEND_HOST_PORT}:8000" \
    -e DYN_DISCOVERY_BACKEND=etcd \
    -e ETCD_ENDPOINTS="http://${ETCD_NAME}:2379" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e HF_MODEL_ID="$HF_MODEL_ID" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "$FRONTEND_IMAGE" >/dev/null

log "frontend reachable on http://localhost:${FRONTEND_HOST_PORT}"
log "tailing frontend logs (Ctrl+C to tear down)"
docker logs -f "$FRONTEND_NAME"
