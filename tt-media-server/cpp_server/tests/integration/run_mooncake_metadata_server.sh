#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Starts the Mooncake Metadata Service used as the discovery mechanism for the
# #4209 migration-worker PoC. This is the registry that lets a sender find a
# receiver by a predefined logical name despite the receiver's OS-assigned
# dynamic transfer-engine port.
#
# Run this ONCE on a host reachable by both the sender and the receiver, then
# point both workers at it with:
#   --metadata http://<THIS_HOST>:<PORT>/metadata
#
# mooncake_master ships in the mooncake-transfer-engine wheel
# (pip install mooncake-transfer-engine==0.3.6.post1); see ../../mooncake/poc1.
#
# Env overrides:
#   HTTP_PORT=N   HTTP metadata server port (default 8080)
#   RPC_PORT=N    master RPC port           (default 50051)
#   BIND_HOST=IP  interface to bind         (default 0.0.0.0)
set -uo pipefail

HTTP_PORT="${HTTP_PORT:-8080}"
RPC_PORT="${RPC_PORT:-50051}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"

if ! command -v mooncake_master >/dev/null 2>&1; then
  echo "ERROR: mooncake_master not found on PATH." >&2
  echo "Install it into the active venv: pip install mooncake-transfer-engine==0.3.6.post1" >&2
  exit 2
fi

echo "Starting Mooncake metadata service:"
echo "  HTTP metadata: http://${BIND_HOST}:${HTTP_PORT}/metadata"
echo "  master RPC:    ${BIND_HOST}:${RPC_PORT}"
echo "Point workers at: --metadata http://<this-host>:${HTTP_PORT}/metadata"

exec mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host="${BIND_HOST}" \
  --http_metadata_server_port="${HTTP_PORT}" \
  --port="${RPC_PORT}"
