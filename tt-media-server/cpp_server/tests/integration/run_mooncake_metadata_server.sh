#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Starts the Mooncake HTTP Metadata Service used as the discovery mechanism for
# the #4209 migration-worker PoC. This is the registry that lets a sender find a
# receiver by a predefined logical name despite the receiver's OS-assigned
# dynamic transfer-engine port.
#
# Run this ONCE on a host reachable by both the sender and the receiver, then
# point both workers at it with:
#   --metadata http://<THIS_HOST>:<PORT>/metadata
#
# IMPORTANT: we launch the wheel's bundled http_metadata_server.py, NOT
# `mooncake_master --enable_http_metadata_server`. The Mooncake C++ transfer
# engine's HTTP metadata client talks to the route `/metadata?key=...` (GET /
# PUT / DELETE), which http_metadata_server.py serves exactly; mooncake_master's
# embedded server uses a different route and returns 404 for those requests.
#
# The server ships in the mooncake-transfer-engine wheel
# (pip install mooncake-transfer-engine==0.3.6.post1; see ../../mooncake/poc1)
# and needs aiohttp (pip install aiohttp).
#
# Env overrides:
#   HTTP_PORT=N   HTTP metadata server port (default 8080)
#   BIND_HOST=IP  interface to bind         (default 0.0.0.0)
#   PYTHON=BIN    interpreter that owns the mooncake wheel (default: auto-detect)
set -uo pipefail

HTTP_PORT="${HTTP_PORT:-8080}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"

# The bundled server must run under the interpreter that owns the mooncake
# wheel. `pip` and the active `python3` are often different environments, so
# resolve the wheel location with pip and run with pip's interpreter unless
# PYTHON is overridden.
PIP_BIN="$(command -v pip || true)"
if [[ -z "${PIP_BIN}" ]]; then
  echo "ERROR: pip not found; cannot locate the mooncake wheel." >&2
  echo "Install it: pip install mooncake-transfer-engine==0.3.6.post1 aiohttp" >&2
  exit 2
fi

if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
else
  PY="$(head -1 "${PIP_BIN}" | sed 's|^#!||')"
  [[ -x "${PY}" ]] || PY="python3"
fi

MOON_DIR="$(pip show mooncake-transfer-engine 2>/dev/null \
  | awk -F': ' '/^Location/{print $2}')/mooncake"
SERVER_PY="${MOON_DIR}/http_metadata_server.py"
if [[ ! -f "${SERVER_PY}" ]]; then
  echo "ERROR: http_metadata_server.py not found at ${SERVER_PY}." >&2
  echo "Install the wheel: pip install mooncake-transfer-engine==0.3.6.post1 aiohttp" >&2
  exit 2
fi

echo "Starting Mooncake HTTP metadata service:"
echo "  server:   ${SERVER_PY}"
echo "  python:   ${PY}"
echo "  endpoint: http://${BIND_HOST}:${HTTP_PORT}/metadata"
echo "Point workers at: --metadata http://<this-host>:${HTTP_PORT}/metadata"

exec "${PY}" "${SERVER_PY}" --host "${BIND_HOST}" --port "${HTTP_PORT}"
