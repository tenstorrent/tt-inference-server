#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 <etcd-image> <cpp-server-image> <frontend-image>

Example:
  $0 quay.io/coreos/etcd:v3.5.13 tt-media-server-cpp:blaze dynamo-frontend

Environment overrides:
  NETWORK_NAME          (default: dynamo-net)
  ETCD_NAME             (default: etcd)
  WORKER_NAME           (default: tt-cpp-worker)
  FRONTEND_NAME         (default: dynamo-frontend)
  FRONTEND_HOST_PORT    (default: 8080)
  MODEL_NAME            (default: tt-cpp-server)
  HF_MODEL_ID           (default: meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN              (default: empty)
  LLM_DEVICE_BACKEND    (default: mock_pipeline)
EOF
    exit 1
}

[[ $# -eq 3 ]] || usage

ETCD_IMAGE="$1"
WORKER_IMAGE="$2"
FRONTEND_IMAGE="$3"

NETWORK_NAME="${NETWORK_NAME:-dynamo-net}"
ETCD_NAME="${ETCD_NAME:-etcd}"
WORKER_NAME="${WORKER_NAME:-tt-cpp-worker}"
FRONTEND_NAME="${FRONTEND_NAME:-dynamo-frontend}"
FRONTEND_HOST_PORT="${FRONTEND_HOST_PORT:-8080}"
MODEL_NAME="${MODEL_NAME:-tt-cpp-server}"
HF_MODEL_ID="${HF_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
LLM_DEVICE_BACKEND="${LLM_DEVICE_BACKEND:-mock_pipeline}"

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
