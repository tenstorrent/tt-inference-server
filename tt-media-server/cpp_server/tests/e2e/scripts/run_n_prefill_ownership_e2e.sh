#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Process-level proof of exclusive N-prefill Kafka ownership (#4795):
#   1. Bring up N prefills + M decodes via local_kv_migration_lab.sh (dry-run)
#   2. Publish one request to the owner partition (migration_cli --partition)
#   3. Assert only the owner prefill logs DryRunMigrationExecutor / ack publish
#   4. Assert non-owner prefills stay idle for that migration_id
#
# Prerequisites:
#   - Kafka up (scripts/dev-kafka.sh up) at KAFKA_BROKERS
#   - Built mooncake_kv_migration_worker
#   - PREFILL_TABLE / DECODE_TABLE .pb files (same as the lab)
#   - confluent-kafka for migration_cli.py
#
# Usage:
#   PREFILL_TABLE=/path/to.pb bash run_n_prefill_ownership_e2e.sh
#   NUM_PREFILL=2 NUM_DECODE=4 LAYER=20 bash run_n_prefill_ownership_e2e.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LAB="${SCRIPT_DIR}/local_kv_migration_lab.sh"
CLI="${CPP_ROOT}/scripts/migration_cli.py"

NUM_PREFILL="${NUM_PREFILL:-2}"
NUM_DECODE="${NUM_DECODE:-2}"
NUM_LAYERS="${NUM_LAYERS:-64}"
# Default target layer maps to partition 1 when N=2 (layers_per_partition=32).
LAYER="${LAYER:-20}"
KAFKA_BROKERS="${KAFKA_BROKERS:-localhost:9092}"
ACK_WAIT_SEC="${ACK_WAIT_SEC:-30}"
STATE_DIR="${STATE_DIR:-/tmp/tt_mc_kv_lab_ownership_e2e}"
KV_MIGRATION_MODE="${KV_MIGRATION_MODE:-dry-run}"
WORKER_BIN="${WORKER_BIN:-${CPP_ROOT}/build/mooncake_kv_migration_worker}"

die() { echo "ERROR: $*" >&2; exit 2; }

layersPerPartition=$(( (NUM_LAYERS + NUM_PREFILL - 1) / NUM_PREFILL ))
ownerPartition=$(( LAYER / layersPerPartition ))
(( ownerPartition >= 0 && ownerPartition < NUM_PREFILL )) || \
  die "LAYER=${LAYER} maps to partition ${ownerPartition}, outside [0, ${NUM_PREFILL})"

migrationId="${MIGRATION_ID:-$(( $(date +%s%N) ))}"
echo "[ownership-e2e] N=${NUM_PREFILL} M=${NUM_DECODE} layer=${LAYER} -> owner=p${ownerPartition}"
echo "[ownership-e2e] brokers=${KAFKA_BROKERS} migration_id=${migrationId}"

cleanup() {
  STATE_DIR="${STATE_DIR}" bash "${LAB}" down >/dev/null 2>&1 || true
}
trap cleanup EXIT

[[ -x "${WORKER_BIN}" ]] || \
  die "mooncake_kv_migration_worker missing; build with --mooncake --kafka"
[[ -f "${CLI}" ]] || die "migration_cli.py missing: ${CLI}"
[[ -f "${PREFILL_TABLE:-}" ]] || die "set PREFILL_TABLE to a readable .pb"

STATE_DIR="${STATE_DIR}" WORKER_BIN="${WORKER_BIN}" \
NUM_PREFILL="${NUM_PREFILL}" NUM_DECODE="${NUM_DECODE}" \
KAFKA_BROKERS="${KAFKA_BROKERS}" KV_MIGRATION_MODE="${KV_MIGRATION_MODE}" \
PREFILL_TABLE="${PREFILL_TABLE}" DECODE_TABLE="${DECODE_TABLE:-${PREFILL_TABLE}}" \
  bash "${LAB}" up --prefill "${NUM_PREFILL}" --decode "${NUM_DECODE}"

# Confirm every prefill pinned and READY.
readyCount=0
for (( i = 0; i < NUM_PREFILL; i++ )); do
  log="${STATE_DIR}/prefill-${i}.log"
  grep -q "kafka_partition=${i}" "${log}" || die "prefill-${i} missing kafka_partition pin in log"
  grep -q "READY:" "${log}" || die "prefill-${i} never reached READY"
  readyCount=$(( readyCount + 1 ))
done
echo "[ownership-e2e] ${readyCount}/${NUM_PREFILL} prefills READY with partition pins"

# Publish exactly one request to the owner partition.
# CLI legacy wire: migration_id becomes kafka_request_id on the worker.
python3 "${CLI}" --brokers "${KAFKA_BROKERS}" produce \
  --partition "${ownerPartition}" \
  --migration-id "${migrationId}" \
  --layer-begin "${LAYER}" --layer-end "$(( LAYER + 1 ))" \
  --src-slot 0 --dst-slot 1 \
  --src-pos-end 16 --dst-pos-end 16 \
  -v

# Wait for owner DryRun execute (StubMigrationExecutor logs kafka_request_id).
deadline=$(( SECONDS + ACK_WAIT_SEC ))
ownerLog="${STATE_DIR}/prefill-${ownerPartition}.log"
while (( SECONDS < deadline )); do
  if grep -q "DryRunMigrationExecutor.*migration_id=${migrationId}" "${ownerLog}" 2>/dev/null; then
    break
  fi
  if grep -q "published ack kafka_request_id=${migrationId}" "${ownerLog}" 2>/dev/null; then
    break
  fi
  sleep 1
done
grep -qE "DryRunMigrationExecutor.*migration_id=${migrationId}|published ack kafka_request_id=${migrationId}" \
  "${ownerLog}" || die "owner did not execute/ack migration_id=${migrationId} within ${ACK_WAIT_SEC}s"

for (( i = 0; i < NUM_PREFILL; i++ )); do
  (( i == ownerPartition )) && continue
  if grep -qE "DryRunMigrationExecutor.*migration_id=${migrationId}|published ack kafka_request_id=${migrationId}" \
      "${STATE_DIR}/prefill-${i}.log" 2>/dev/null; then
    die "non-owner prefill-${i} executed migration_id=${migrationId}"
  fi
done

echo "[ownership-e2e] PASS: only prefill-${ownerPartition} executed migration_id=${migrationId}"
