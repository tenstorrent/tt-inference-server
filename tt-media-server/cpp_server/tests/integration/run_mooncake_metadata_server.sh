#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Starts the Mooncake HTTP metadata service (the #4209 discovery registry) on a
# host both workers can reach: --metadata http://<THIS_HOST>:<PORT>/metadata
#
# Launches the wheel's http_metadata_server.py, NOT mooncake_master: the C++
# client uses the /metadata?key=... route, which only the Python server serves
# (mooncake_master's embedded server returns 404 for it).
#
# Env: HTTP_PORT (8080), BIND_HOST (0.0.0.0), PYTHON (wheel's interpreter).
set -uo pipefail

HTTP_PORT="${HTTP_PORT:-8080}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"

# pip and python3 are often different environments; run under pip's interpreter.
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

# Fall back to the in-tree wheel when the package isn't pip-installed (e.g. a
# Mooncake build that vendors the wheel without installing it).
if [[ ! -f "${SERVER_PY}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_SERVER_PY="${SCRIPT_DIR}/../../third_party/Mooncake/mooncake-wheel/mooncake/http_metadata_server.py"
  [[ -f "${REPO_SERVER_PY}" ]] && SERVER_PY="${REPO_SERVER_PY}"
fi

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
