#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# MooncakeKafkaMigration single-host e2e test. Builds on the #4294 discovery
# test but adds the Kafka request path:
#
#   1. Verify the dev Kafka broker is reachable and topics exist (the workers
#      do not auto-create topics; that's the responsibility of
#      `scripts/migration_cli.py setup`).
#   2. Start the Mooncake metadata service.
#   3. Launch ${NUM_PREFILL} prefill workers (with Kafka) via one mpirun and
#      ${NUM_DECODE} decode workers (--no-kafka) via a second mpirun. Wait for
#      all ${NUM_PREFILL} + ${NUM_DECODE} "CONNECTED to" lines and all
#      ${NUM_PREFILL} "entering KV-migration loop" lines.
#   4. Produce one migration request (unique migration_id per run) and verify
#      ${NUM_PREFILL} SUCCESSFUL acks come back within ACK_TIMEOUT_SEC.
#   5. Tear everything down.
#
# Each prefill worker is launched in its own Kafka consumer group (see
# migration_worker_rank_launch.sh), so a single request fans out to all
# prefills — one ack per worker is the contract this test asserts.
#
# Env: WORKER_BIN (./build/bringup_mooncake_worker), KAFKA_BROKERS (kafka:9092),
# METADATA (auto-started if unset), HTTP_PORT (18083),
# HOST_DRAM_BYTES (1 MiB), DISCOVERY_TIMEOUT_SEC (60), ACK_TIMEOUT_SEC (30),
# MC_BIND_ADDRESS (127.0.0.1), MIGRATION_CLI_VENV (auto-detected:
# cpp_server/.venv if present).
set -uo pipefail

NUM_PREFILL="${NUM_PREFILL:-4}"
NUM_DECODE="${NUM_DECODE:-16}"
WORKER_BIN="${WORKER_BIN:-./build/bringup_mooncake_worker}"
HTTP_PORT="${HTTP_PORT:-18083}"
HOST_DRAM_BYTES="${HOST_DRAM_BYTES:-1048576}"
DISCOVERY_TIMEOUT_SEC="${DISCOVERY_TIMEOUT_SEC:-60}"
ACK_TIMEOUT_SEC="${ACK_TIMEOUT_SEC:-30}"
MC_BIND_ADDRESS="${MC_BIND_ADDRESS:-127.0.0.1}"
KAFKA_BROKERS="${KAFKA_BROKERS:-kafka:9092}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK_LAUNCH="${SCRIPT_DIR}/migration_worker_rank_launch.sh"
META_SERVER="${SCRIPT_DIR}/../../integration/run_mooncake_metadata_server.sh"
ACK_COUNTER="${SCRIPT_DIR}/produce_and_count_acks.py"
PREFILL_LOG="${PREFILL_LOG:-/tmp/tt_mc_mpi_prefill.log}"
DECODE_LOG="${DECODE_LOG:-/tmp/tt_mc_mpi_decode.log}"
META_LOG="${META_LOG:-/tmp/tt_mc_metadata_kafka_mpi.log}"
ACK_LOG="${ACK_LOG:-/tmp/tt_mc_ack_counter.log}"

# Pick a Python interpreter that has confluent-kafka available. Prefer the
# project's .venv (created by `scripts/setup-migration-cli.sh`) so the test
# stays hermetic on dev machines.
PYTHON=""
if [[ -n "${MIGRATION_CLI_VENV:-}" && -x "${MIGRATION_CLI_VENV}/bin/python" ]]; then
  PYTHON="${MIGRATION_CLI_VENV}/bin/python"
elif [[ -x "${SCRIPT_DIR}/../../../.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/../../../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
fi

# --- preflight ---------------------------------------------------------------

if [[ ! -x "${WORKER_BIN}" ]]; then
  echo "ERROR: ${WORKER_BIN} not found/executable." >&2
  echo "Build it: ./build.sh --blaze --kafka --mooncake (Kafka+Mooncake required)" >&2
  exit 2
fi
if ! command -v mpirun >/dev/null 2>&1; then
  echo "ERROR: mpirun not found; install Open MPI to run this test." >&2
  exit 2
fi
if [[ ! -x "${ACK_COUNTER}" ]]; then
  echo "ERROR: ${ACK_COUNTER} not found/executable." >&2
  exit 2
fi
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: no python3 available." >&2
  exit 2
fi
if ! "${PYTHON}" -c "import confluent_kafka" 2>/dev/null; then
  echo "ERROR: confluent_kafka not importable from ${PYTHON}." >&2
  echo "Hint: bash scripts/setup-migration-cli.sh, or set MIGRATION_CLI_VENV." >&2
  exit 2
fi

echo "Probing Kafka broker at ${KAFKA_BROKERS}..."
if ! KAFKA_BROKERS="${KAFKA_BROKERS}" "${PYTHON}" - <<'PY' >/dev/null 2>&1; then
import os, sys
from confluent_kafka.admin import AdminClient
admin = AdminClient({"bootstrap.servers": os.environ["KAFKA_BROKERS"]})
md = admin.list_topics(timeout=5)
required = {"kv-migration-requests", "kv-migration-acks"}
missing = required - set(md.topics)
if missing:
    print(f"missing topics: {sorted(missing)}", file=sys.stderr)
    sys.exit(2)
PY
  echo "ERROR: Kafka at ${KAFKA_BROKERS} not reachable or topics missing." >&2
  echo "Hint: bash scripts/dev-kafka.sh up && python scripts/migration_cli.py setup" >&2
  exit 2
fi
echo "Kafka topics OK."

# --- cleanup wiring ----------------------------------------------------------

meta_pid=""
prefill_pid=""
decode_pid=""
cleanup() {
  [[ -n "${prefill_pid}" ]] && kill "${prefill_pid}" 2>/dev/null
  [[ -n "${decode_pid}" ]] && kill "${decode_pid}" 2>/dev/null
  [[ -n "${meta_pid}" ]] && kill "${meta_pid}" 2>/dev/null
  pkill -f "bringup_mooncake_worker --metadata" 2>/dev/null || true
}
trap cleanup EXIT

# --- metadata service --------------------------------------------------------

probe_metadata() {
  "${PYTHON}" - "$1" <<'PY' 2>/dev/null
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
    "${META_SERVER}" >"${META_LOG}" 2>&1 &
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

# --- launch workers ----------------------------------------------------------

: >"${PREFILL_LOG}"
: >"${DECODE_LOG}"

common_env=(
  MC_TCP_BIND_ADDRESS="${MC_BIND_ADDRESS}"
  WORKER_BIN="${WORKER_BIN}"
  METADATA="${META_URI}"
  HOST_DRAM_BYTES="${HOST_DRAM_BYTES}"
  DISCOVERY_TIMEOUT_SEC="${DISCOVERY_TIMEOUT_SEC}"
  NUM_PREFILL="${NUM_PREFILL}"
  NUM_DECODE="${NUM_DECODE}"
  KAFKA_BROKERS="${KAFKA_BROKERS}"
)

echo "Launching ${NUM_PREFILL} prefill workers (Kafka) via mpirun..."
env "${common_env[@]}" WORKER_ROLE=prefill \
  mpirun --oversubscribe --tag-output -np "${NUM_PREFILL}" \
    bash "${RANK_LAUNCH}" >"${PREFILL_LOG}" 2>&1 &
prefill_pid=$!

echo "Launching ${NUM_DECODE} decode workers (--no-kafka) via mpirun..."
env "${common_env[@]}" WORKER_ROLE=decode \
  mpirun --oversubscribe --tag-output -np "${NUM_DECODE}" \
    bash "${RANK_LAUNCH}" >"${DECODE_LOG}" 2>&1 &
decode_pid=$!

# --- wait for mesh -----------------------------------------------------------

readonly NUM_WORKERS=$(( NUM_PREFILL + NUM_DECODE ))
echo "Waiting up to ${DISCOVERY_TIMEOUT_SEC}s for ${NUM_WORKERS} CONNECTED + ${NUM_PREFILL} KV-migration-loop lines..."

count_lines() { grep -c "$1" "$2" 2>/dev/null || true; }

deadline=$(( SECONDS + DISCOVERY_TIMEOUT_SEC + 10 ))
ready=0
while (( SECONDS < deadline )); do
  prefill_connected="$(count_lines 'CONNECTED to' "${PREFILL_LOG}")"
  decode_connected="$(count_lines 'CONNECTED to' "${DECODE_LOG}")"
  prefill_kafka_ready="$(count_lines 'entering KV-migration loop' "${PREFILL_LOG}")"
  prefill_connected="${prefill_connected:-0}"
  decode_connected="${decode_connected:-0}"
  prefill_kafka_ready="${prefill_kafka_ready:-0}"
  total_connected=$(( prefill_connected + decode_connected ))
  if (( total_connected >= NUM_WORKERS && prefill_kafka_ready >= NUM_PREFILL )); then
    ready=1
    break
  fi
  if ! kill -0 "${prefill_pid}" 2>/dev/null || ! kill -0 "${decode_pid}" 2>/dev/null; then
    echo "ERROR: one mpirun exited before mesh was up." >&2
    break
  fi
  sleep 1
done

prefill_connected="$(count_lines 'CONNECTED to' "${PREFILL_LOG}")"
decode_connected="$(count_lines 'CONNECTED to' "${DECODE_LOG}")"
prefill_kafka_ready="$(count_lines 'entering KV-migration loop' "${PREFILL_LOG}")"
prefill_connected="${prefill_connected:-0}"
decode_connected="${decode_connected:-0}"
prefill_kafka_ready="${prefill_kafka_ready:-0}"
total_connected=$(( prefill_connected + decode_connected ))
echo "----------------------------------------"
echo "connected: prefill=${prefill_connected}/${NUM_PREFILL} decode=${decode_connected}/${NUM_DECODE} total=${total_connected}/${NUM_WORKERS}"
echo "prefill in KV-migration loop: ${prefill_kafka_ready}/${NUM_PREFILL}"

if (( ready != 1 )); then
  echo "RESULT: FAIL (mesh not ready)"
  echo "--- last 40 lines of ${PREFILL_LOG} ---" >&2
  tail -40 "${PREFILL_LOG}" >&2 || true
  echo "--- last 40 lines of ${DECODE_LOG} ---" >&2
  tail -40 "${DECODE_LOG}" >&2 || true
  exit 1
fi

# --- exercise Kafka path -----------------------------------------------------

# nanoseconds-since-epoch gives us a fresh migration_id per run even when the
# test is re-run quickly; uniqueness matters because the ack counter matches
# by migration_id alone.
migration_id=$(date +%s%N)
echo "Producing 1 request with migration_id=${migration_id}, expecting ${NUM_PREFILL} acks..."
if "${PYTHON}" "${ACK_COUNTER}" \
    --brokers "${KAFKA_BROKERS}" \
    --migration-id "${migration_id}" \
    --expected-acks "${NUM_PREFILL}" \
    --timeout-sec "${ACK_TIMEOUT_SEC}" 2>&1 | tee "${ACK_LOG}"; then
  echo "RESULT: PASS"
  exit 0
fi

echo "RESULT: FAIL"
echo "--- last 40 lines of ${PREFILL_LOG} ---" >&2
tail -40 "${PREFILL_LOG}" >&2 || true
exit 1
