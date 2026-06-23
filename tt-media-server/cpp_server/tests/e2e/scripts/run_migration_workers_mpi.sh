#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# #4294 single-host MPI integration test. Brings up the metadata service, then
# launches 4 prefill + 16 decode bringup_mooncake_worker processes at once via
# mpirun (rank->identity mapping in migration_worker_rank_launch.sh) and checks
# that all 20 discover their peers within DISCOVERY_TIMEOUT_SEC. Success is 20
# "CONNECTED to N peers" lines; the workers hold until SIGTERM, so we tear them
# down once the mesh is verified.
#
# Env: WORKER_BIN (./build/bringup_mooncake_worker), METADATA (auto-started if
# unset), HTTP_PORT (18080), HOST_DRAM_BYTES (1 MiB), DISCOVERY_TIMEOUT_SEC (60),
# MC_BIND_ADDRESS (127.0.0.1).
set -uo pipefail

readonly NUM_WORKERS=20
WORKER_BIN="${WORKER_BIN:-./build/bringup_mooncake_worker}"
HTTP_PORT="${HTTP_PORT:-18080}"
HOST_DRAM_BYTES="${HOST_DRAM_BYTES:-1048576}"
DISCOVERY_TIMEOUT_SEC="${DISCOVERY_TIMEOUT_SEC:-60}"
MC_BIND_ADDRESS="${MC_BIND_ADDRESS:-127.0.0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK_LAUNCH="${SCRIPT_DIR}/migration_worker_rank_launch.sh"
MPI_LOG="${MPI_LOG:-/tmp/tt_mc_mpi_workers.log}"
META_LOG="${META_LOG:-/tmp/tt_mc_metadata_mpi.log}"

if [[ ! -x "${WORKER_BIN}" ]]; then
  echo "ERROR: ${WORKER_BIN} not found/executable." >&2
  echo "Build it (needs Mooncake): cmake --build build --target bringup_mooncake_worker" >&2
  exit 2
fi
if ! command -v mpirun >/dev/null 2>&1; then
  echo "ERROR: mpirun not found; install Open MPI to run this test." >&2
  exit 2
fi

meta_pid=""
mpi_pid=""
cleanup() {
  [[ -n "${mpi_pid}" ]] && kill "${mpi_pid}" 2>/dev/null
  [[ -n "${meta_pid}" ]] && kill "${meta_pid}" 2>/dev/null
  # bringup workers hold until SIGTERM; sweep stragglers. Match the full worker
  # invocation (binary + --metadata) so we never hit a launcher shell whose
  # argv merely mentions the binary path.
  pkill -f "bringup_mooncake_worker --metadata" 2>/dev/null || true
}
trap cleanup EXIT

probe_metadata() {
  python3 - "$1" <<'PY' 2>/dev/null
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
  echo "Starting metadata service on ${MC_BIND_ADDRESS}:${HTTP_PORT}..."
  HTTP_PORT="${HTTP_PORT}" BIND_HOST="${MC_BIND_ADDRESS}" \
    "${SCRIPT_DIR}/run_mooncake_metadata_server.sh" >"${META_LOG}" 2>&1 &
  meta_pid=$!
  META_URI="http://${MC_BIND_ADDRESS}:${HTTP_PORT}/metadata"
  ready=0
  for _ in $(seq 1 20); do
    if probe_metadata "${META_URI}"; then ready=1; break; fi
    sleep 0.5
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "ERROR: metadata service not ready at ${META_URI} (port in use?)" >&2
    cat "${META_LOG}" >&2 || true
    exit 1
  fi
  echo "Metadata service ready at ${META_URI}"
fi

echo "Launching ${NUM_WORKERS} workers (4 prefill + 16 decode) via mpirun..."
: >"${MPI_LOG}"
# --oversubscribe: 20 ranks on one CI host with fewer cores.
# --tag-output: prefix each line with its rank for debuggable logs.
MC_TCP_BIND_ADDRESS="${MC_BIND_ADDRESS}" \
WORKER_BIN="${WORKER_BIN}" \
METADATA="${META_URI}" \
HOST_DRAM_BYTES="${HOST_DRAM_BYTES}" \
DISCOVERY_TIMEOUT_SEC="${DISCOVERY_TIMEOUT_SEC}" \
  mpirun --oversubscribe --tag-output -np "${NUM_WORKERS}" \
    bash "${RANK_LAUNCH}" >"${MPI_LOG}" 2>&1 &
mpi_pid=$!

echo "Waiting up to ${DISCOVERY_TIMEOUT_SEC}s for all ${NUM_WORKERS} workers to connect..."
connected=0
deadline=$(( SECONDS + DISCOVERY_TIMEOUT_SEC + 5 ))
while (( SECONDS < deadline )); do
  connected="$(grep -c "CONNECTED to" "${MPI_LOG}" 2>/dev/null || true)"
  connected="${connected:-0}"
  if (( connected >= NUM_WORKERS )); then break; fi
  # If mpirun already exited, no more output is coming.
  if ! kill -0 "${mpi_pid}" 2>/dev/null; then break; fi
  sleep 1
done

connected="$(grep -c "CONNECTED to" "${MPI_LOG}" 2>/dev/null || true)"
connected="${connected:-0}"
echo "----------------------------------------"
echo "workers connected: ${connected}/${NUM_WORKERS}"
if (( connected >= NUM_WORKERS )); then
  echo "RESULT: PASS"
  exit 0
fi
echo "RESULT: FAIL"
echo "--- last 40 lines of ${MPI_LOG} ---" >&2
tail -40 "${MPI_LOG}" >&2 || true
exit 1
