#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Single-container dev Kafka broker for cpp_server.
#
# Brings up apache/kafka:4.0.0 in single-node KRaft mode (combined broker +
# controller) attached to the tt_net network. Cluster config is declared in
# kafka-server.properties (sibling of this script), mounted into the image's
# user-config layer.
#
# Topic provisioning is intentionally NOT done here -- run
#   python migration_cli.py setup
# which uses the Kafka AdminClient over TCP. This mirrors the production
# pattern (admin SDK called at deploy time) and avoids `docker exec`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER="${KAFKA_CONTAINER:-kafka}"
IMAGE="${KAFKA_IMAGE:-apache/kafka:4.0.0}"
NETWORK="${KAFKA_NETWORK:-tt_net}"
PROPS_FILE="${KAFKA_PROPS:-$SCRIPT_DIR/kafka-server.properties}"
READY_TIMEOUT_S="${KAFKA_READY_TIMEOUT_S:-30}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  up               Start kafka if not already running and wait until ready
  down             Remove the kafka container
  restart          down + up
  status           Show container status and topic list
  logs [-f]        Show kafka logs (pass any extra args through to docker logs)
  shell            Open a bash shell inside the kafka container

Environment overrides:
  KAFKA_CONTAINER         (default: $CONTAINER)
  KAFKA_IMAGE             (default: $IMAGE)
  KAFKA_NETWORK           (default: $NETWORK)
  KAFKA_PROPS             (default: $PROPS_FILE)
  KAFKA_READY_TIMEOUT_S   (default: $READY_TIMEOUT_S)
EOF
}

is_running() {
  [ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" = "true" ]
}

ensure_network() {
  if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
    echo "Creating docker network: $NETWORK"
    docker network create "$NETWORK" >/dev/null
  fi
}

wait_ready() {
  echo "Waiting for broker (timeout=${READY_TIMEOUT_S}s)..."
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  while [ $SECONDS -lt $deadline ]; do
    if docker exec "$CONTAINER" /opt/kafka/bin/kafka-broker-api-versions.sh \
        --bootstrap-server kafka:9092 >/dev/null 2>&1; then
      echo "Broker ready."
      return 0
    fi
    sleep 1
  done
  echo "Broker did not become ready in ${READY_TIMEOUT_S}s. Recent logs:" >&2
  docker logs --tail 40 "$CONTAINER" >&2
  return 1
}

cmd_up() {
  if is_running; then
    echo "Container '$CONTAINER' already running."
    return 0
  fi
  if [ ! -r "$PROPS_FILE" ]; then
    echo "Properties file not readable: $PROPS_FILE" >&2
    exit 1
  fi
  ensure_network
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d \
    --name "$CONTAINER" \
    --network "$NETWORK" \
    --restart unless-stopped \
    -v "$PROPS_FILE:/mnt/shared/config/server.properties:ro" \
    "$IMAGE" >/dev/null
  echo "Started container '$CONTAINER' (image=$IMAGE network=$NETWORK)"
  wait_ready
  cat <<EOF

Next steps:
  python $SCRIPT_DIR/migration_cli.py setup    # create app topics
  python $SCRIPT_DIR/migration_cli.py status   # confirm
EOF
}

cmd_down() {
  if docker rm -f "$CONTAINER" >/dev/null 2>&1; then
    echo "Removed container '$CONTAINER'."
  else
    echo "No container '$CONTAINER' to remove."
  fi
}

cmd_restart() { cmd_down; cmd_up; }

cmd_status() {
  echo "=== container ==="
  docker ps -a --filter "name=^${CONTAINER}$" \
    --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
  if is_running; then
    echo ""
    echo "=== topics ==="
    docker exec "$CONTAINER" /opt/kafka/bin/kafka-topics.sh \
      --bootstrap-server kafka:9092 --list
  fi
}

cmd_logs() { docker logs "$@" "$CONTAINER"; }

cmd_shell() { docker exec -it "$CONTAINER" bash; }

cmd="${1:-}"; shift || true
case "$cmd" in
  up)      cmd_up ;;
  down)    cmd_down ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  logs)    cmd_logs "$@" ;;
  shell)   cmd_shell ;;
  ""|-h|--help) usage ;;
  *)       usage; exit 1 ;;
esac
