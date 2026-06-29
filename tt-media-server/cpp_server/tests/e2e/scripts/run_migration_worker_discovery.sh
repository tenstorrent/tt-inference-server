#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# #4209 single-host smoke test: starts the metadata service (unless METADATA is
# set) and runs a receiver + sender on 127.0.0.1, reporting PASS/FAIL. The
# two-host runbook and build requirements (USE_HTTP=ON + libcurl) are in
# src/transport/README.md.
#
# Env: METADATA (use an existing service), HTTP_PORT (8080), BYTES (65536),
# TIMEOUT_SEC (30), RECV_NAME/SEND_NAME (kv-receiver-0/kv-sender-0).
set -uo pipefail

BIN="${1:-./build/migration_worker_discovery}"
BYTES="${BYTES:-65536}"
TIMEOUT_SEC="${TIMEOUT_SEC:-30}"
RECV_NAME="${RECV_NAME:-kv-receiver-0}"
SEND_NAME="${SEND_NAME:-kv-sender-0}"
HTTP_PORT="${HTTP_PORT:-8080}"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: ${BIN} not found/executable." >&2
  echo "Build the discovery target (needs Mooncake + USE_HTTP + libcurl):" >&2
  echo "  cmake --build build --target migration_worker_discovery" >&2
  echo "See src/transport/README.md (#4209) for the full build requirements." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Metadata service launcher is infrastructure, kept under tests/integration/.
META_SERVER="${SCRIPT_DIR}/../../integration/run_mooncake_metadata_server.sh"
META_LOG="${META_LOG:-/tmp/tt_mc_metadata.log}"
meta_pid=""
cleanup() { [[ -n "${meta_pid}" ]] && kill "${meta_pid}" 2>/dev/null; }
trap cleanup EXIT

# Probe with a real PUT (the C++ client's contract) via stdlib urllib, so we
# don't depend on a curl executable.
probe_metadata() {
  local uri="$1"
  python3 - "$uri" <<'PY' 2>/dev/null
import sys, urllib.request as u
try:
    r = u.urlopen(u.Request(sys.argv[1] + "?key=__probe__", data=b"{}",
                            method="PUT"), timeout=2)
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

if [[ -n "${METADATA:-}" ]]; then
  META_URI="${METADATA}"
  echo "Using existing metadata service: ${META_URI}"
else
  echo "Auto-starting HTTP metadata service on 127.0.0.1:${HTTP_PORT}..."
  HTTP_PORT="${HTTP_PORT}" BIND_HOST=127.0.0.1 \
    "${META_SERVER}" >"${META_LOG}" 2>&1 &
  meta_pid=$!
  META_URI="http://127.0.0.1:${HTTP_PORT}/metadata"
  # Wait until it answers a PUT; binding can fail (e.g. port in use).
  ready=0
  for _ in $(seq 1 20); do
    if probe_metadata "${META_URI}"; then ready=1; break; fi
    sleep 0.5
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "ERROR: metadata service did not become ready at ${META_URI}" >&2
    echo "  (is port ${HTTP_PORT} already in use? try HTTP_PORT=18080)" >&2
    echo "  log: ${META_LOG}" >&2
    cat "${META_LOG}" >&2 || true
    exit 1
  fi
  echo "Metadata service ready at ${META_URI}"
fi

base=(--metadata "${META_URI}" --bytes "${BYTES}" --timeout-sec "${TIMEOUT_SEC}")

echo "Launching receiver (name=${RECV_NAME})..."
"${BIN}" --role receiver --name "${RECV_NAME}" "${base[@]}" &
recv_pid=$!

# Let the receiver register before the sender looks it up.
sleep 1

echo "Launching sender (name=${SEND_NAME} -> peer ${RECV_NAME})..."
"${BIN}" --role sender --name "${SEND_NAME}" --peer "${RECV_NAME}" "${base[@]}"
send_rc=$?

wait "${recv_pid}"
recv_rc=$?

echo "----------------------------------------"
echo "sender exit=${send_rc}  receiver exit=${recv_rc}"
if [[ ${send_rc} -eq 0 && ${recv_rc} -eq 0 ]]; then
  echo "RESULT: PASS"
  exit 0
fi
echo "RESULT: FAIL"
exit 1
