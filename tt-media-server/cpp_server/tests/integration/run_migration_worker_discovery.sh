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
# Requires a Mooncake-enabled build (./build.sh --mooncake), which produces
# build/migration_worker_discovery, and a running metadata service
# (tests/integration/run_mooncake_metadata_server.sh).
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
  echo "ERROR: ${BIN} not found/executable. Build with: ./build.sh --mooncake" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
master_pid=""
cleanup() { [[ -n "${master_pid}" ]] && kill "${master_pid}" 2>/dev/null; }
trap cleanup EXIT

# Use an existing metadata service, or auto-start one for the smoke test.
if [[ -n "${METADATA:-}" ]]; then
  META_URI="${METADATA}"
  echo "Using existing metadata service: ${META_URI}"
else
  if ! command -v mooncake_master >/dev/null 2>&1; then
    echo "ERROR: mooncake_master not on PATH and METADATA not set." >&2
    echo "Either set METADATA=http://host:port/metadata or install mooncake_master" >&2
    echo "(pip install mooncake-transfer-engine==0.3.6.post1)." >&2
    exit 2
  fi
  echo "Auto-starting metadata service on 127.0.0.1:${HTTP_PORT}..."
  HTTP_PORT="${HTTP_PORT}" BIND_HOST=127.0.0.1 \
    "${SCRIPT_DIR}/run_mooncake_metadata_server.sh" >/tmp/tt_mc_master.log 2>&1 &
  master_pid=$!
  META_URI="http://127.0.0.1:${HTTP_PORT}/metadata"
  sleep 2  # give the master time to bind its HTTP port
  if ! kill -0 "${master_pid}" 2>/dev/null; then
    echo "ERROR: metadata service failed to start; see /tmp/tt_mc_master.log" >&2
    cat /tmp/tt_mc_master.log >&2 || true
    exit 1
  fi
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
