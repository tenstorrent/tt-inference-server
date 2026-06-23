#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Start minimal Dynamo infrastructure (etcd + frontend) for CI tests.
# Usage: ./start_dynamo.sh [--stop]
#
# Reuses the same container images as dynamo_frontend/deploy.sh but only starts
# the infrastructure components (etcd + frontend). Tests spawn their own
# TestServer backends that register with etcd.

set -euo pipefail

NETWORK_NAME="dynamo-ci"
ETCD_NAME="dynamo-ci-etcd"
FRONTEND_NAME="dynamo-ci-frontend"
FRONTEND_HOST_PORT="${DYNAMO_PORT:-8080}"

# Same images as dynamo_frontend/deploy.sh
ETCD_IMAGE="quay.io/coreos/etcd:v3.5.13"
FRONTEND_IMAGE="${DYNAMO_FRONTEND_IMAGE:-ghcr.io/tenstorrent/tt-shield/tt-dynamo-frontend:ef76035_20260605_091917}"

log() { printf '[dynamo-ci] %s\n' "$*"; }
die() { printf '[dynamo-ci] ERROR: %s\n' "$*" >&2; exit 1; }

stop_dynamo() {
    log "stopping Dynamo infrastructure"
    docker rm -f "$FRONTEND_NAME" "$ETCD_NAME" 2>/dev/null || true
    docker network rm "$NETWORK_NAME" 2>/dev/null || true
    log "stopped"
}

start_dynamo() {
    # Check if already running
    if docker ps --format '{{.Names}}' | grep -q "^${FRONTEND_NAME}\$"; then
        log "Dynamo already running"
        return 0
    fi

    # Clean up any stale containers
    docker rm -f "$FRONTEND_NAME" "$ETCD_NAME" 2>/dev/null || true

    # Create network if needed
    docker network inspect "$NETWORK_NAME" >/dev/null 2>&1 \
        || { log "creating network $NETWORK_NAME"; docker network create "$NETWORK_NAME" >/dev/null; }

    # Start etcd
    log "starting etcd ($ETCD_IMAGE)"
    docker run -d --name "$ETCD_NAME" --network "$NETWORK_NAME" -p 2379:2379 \
        "$ETCD_IMAGE" /usr/local/bin/etcd --name dyn-etcd \
            --advertise-client-urls http://0.0.0.0:2379 \
            --listen-client-urls http://0.0.0.0:2379 >/dev/null

    # Wait for etcd to be healthy
    log "waiting for etcd"
    for _ in $(seq 1 30); do
        if docker exec "$ETCD_NAME" etcdctl endpoint health >/dev/null 2>&1; then
            log "etcd healthy"
            break
        fi
        sleep 1
    done
    docker exec "$ETCD_NAME" etcdctl endpoint health >/dev/null 2>&1 \
        || { docker logs --tail 50 "$ETCD_NAME" >&2 || true; die "etcd never became healthy"; }

    # Start frontend
    log "starting frontend ($FRONTEND_IMAGE)"
    docker run -d --name "$FRONTEND_NAME" --network "$NETWORK_NAME" -p "${FRONTEND_HOST_PORT}:8000" \
        -e DYN_DISCOVERY_BACKEND=etcd \
        -e ETCD_ENDPOINTS="http://${ETCD_NAME}:2379" \
        -e MODEL_NAME="tt-cpp-server" \
        -e DYN_CHAT_PROCESSOR=dynamo \
        -e DYN_TOKENIZER=fastokens \
        -e ROUTER_MODE=kv \
        "$FRONTEND_IMAGE" >/dev/null

    # Wait for frontend to be ready (check if container is running)
    log "waiting for frontend"
    for _ in $(seq 1 30); do
        if ! docker ps --format '{{.Names}}' | grep -q "^${FRONTEND_NAME}\$"; then
            docker logs --tail 50 "$FRONTEND_NAME" >&2 || true
            die "frontend exited during startup"
        fi
        # Try health endpoint if available
        if curl -sf "http://127.0.0.1:${FRONTEND_HOST_PORT}/health" >/dev/null 2>&1; then
            log "frontend ready on http://127.0.0.1:${FRONTEND_HOST_PORT}"
            return 0
        fi
        sleep 1
    done

    # Frontend might not have /health, just check if it's still running
    if docker ps --format '{{.Names}}' | grep -q "^${FRONTEND_NAME}\$"; then
        log "frontend running on http://127.0.0.1:${FRONTEND_HOST_PORT}"
        return 0
    fi

    docker logs --tail 50 "$FRONTEND_NAME" >&2 || true
    die "frontend never became ready"
}

case "${1:-start}" in
    --stop|stop)
        stop_dynamo
        ;;
    --start|start|"")
        start_dynamo
        ;;
    *)
        echo "Usage: $0 [--start|--stop]" >&2
        exit 1
        ;;
esac
