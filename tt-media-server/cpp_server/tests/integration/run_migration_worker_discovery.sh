#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# #4209 migration-worker discovery PoC driver.
#
# Validates that the Mooncake Metadata Service is sufficient for two transfer
# engines to find each other by a PREDEFINED LOGICAL NAME, with the actual
# transfer port left dynamic (OS-assigned). Host RAM only, no device memory.
#
# Requires a Mooncake-enabled build WITH the HTTP metadata plugin
# (USE_HTTP=ON + libcurl) producing build/migration_worker_discovery, and a
# running metadata service (tests/integration/run_mooncake_metadata_server.sh).
# See src/transport/README.md (#4209 section) for the build requirements: the
# default tt-llm-engine build forces USE_HTTP=OFF, so http:// discovery is
# unavailable until it is re-enabled.
#
# ── Two-host run (the real PoC) ────────────────────────────────────────────
#   # On the metadata host (reachable by both):
#   tests/integration/run_mooncake_metadata_server.sh
#
#   # On the receiver host:
#   build/migration_worker_discovery --role receiver \
#     --metadata http://META_HOST:8080/metadata --name kv-receiver-0 --bytes 1048576
#
#   # On the sender host:
#   build/migration_worker_discovery --role sender \
#     --metadata http://META_HOST:8080/metadata --name kv-sender-0 \
#     --peer kv-receiver-0 --bytes 1048576
#
# ── Single-host smoke (this script) ────────────────────────────────────────
#   Starts the metadata service (unless METADATA is given) and runs both workers
#   on 127.0.0.1, then reports PASS/FAIL.
#
# Usage:
#   tests/integration/run_migration_worker_discovery.sh [path-to-binary]
#
# Env overrides:
#   METADATA=URI        use an EXISTING metadata service instead of starting one
#                       (e.g. http://META_HOST:8080/metadata)
#   HTTP_PORT=N         metadata HTTP port when auto-starting (default 8080)
#   BYTES=N             tensor size (default 65536)
#   TIMEOUT_SEC=S       (default 30)
#   RECV_NAME/SEND_NAME logical names (default kv-receiver-0 / kv-sender-0)
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
META_LOG="${META_LOG:-/tmp/tt_mc_metadata.log}"
meta_pid=""
cleanup() { [[ -n "${meta_pid}" ]] && kill "${meta_pid}" 2>/dev/null; }
trap cleanup EXIT

# Probe the metadata endpoint with a real PUT (the C++ client's contract), using
# stdlib urllib so we don't depend on a curl executable being installed.
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

# Use an existing metadata service, or auto-start one for the smoke test.
if [[ -n "${METADATA:-}" ]]; then
  META_URI="${METADATA}"
  echo "Using existing metadata service: ${META_URI}"
else
  echo "Auto-starting HTTP metadata service on 127.0.0.1:${HTTP_PORT}..."
  HTTP_PORT="${HTTP_PORT}" BIND_HOST=127.0.0.1 \
    "${SCRIPT_DIR}/run_mooncake_metadata_server.sh" >"${META_LOG}" 2>&1 &
  meta_pid=$!
  META_URI="http://127.0.0.1:${HTTP_PORT}/metadata"
  # Wait until it actually answers a PUT (binding can fail, e.g. port in use).
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

# Give the receiver a moment to register its segment in the metadata service.
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
