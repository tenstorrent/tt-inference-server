#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Spins up 3 MigrationWorker consumer instances and 1 dummy producer.
# Messages are distributed across workers by Kafka consumer group rebalancing.
#
# Usage:
#   tests/integration/run_migration_workers.sh [build-dir]
#
# Env overrides:
#   KAFKA_BROKERS             (default: localhost:9092)
#   KAFKA_OFFLOAD_TOPIC_NAME  (default: offload_requests)
#   KAFKA_GROUP_ID            (default: tt-migration-workers)
#   NUM_WORKERS               (default: 3)
#   PRODUCER_COUNT            (default: 20, number of messages to produce)
#   PRODUCER_INTERVAL_MS      (default: 200)
#   WORKER_BASE_PORT          (default: 9100, each worker gets basePort + i)
#
# Prerequisites:
#   - Kafka running at KAFKA_BROKERS (e.g. docker compose up kafka)
#   - Build with -DKAFKA_ENABLED=ON (./build.sh --kafka)
#
# Example:
#   # Start Kafka first, then:
#   tests/integration/run_migration_workers.sh ./build

set -euo pipefail

BUILD_DIR="${1:-./build}"
CONSUMER_BIN="${BUILD_DIR}/tt_consumer"
PRODUCER_BIN="${BUILD_DIR}/migration_producer_dummy"

NUM_WORKERS="${NUM_WORKERS:-3}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-9100}"
PRODUCER_COUNT="${PRODUCER_COUNT:-20}"
PRODUCER_INTERVAL_MS="${PRODUCER_INTERVAL_MS:-200}"

export KAFKA_BROKERS="${KAFKA_BROKERS:-localhost:9092}"
export KAFKA_OFFLOAD_TOPIC_NAME="${KAFKA_OFFLOAD_TOPIC_NAME:-offload_requests}"
export KAFKA_GROUP_ID="${KAFKA_GROUP_ID:-tt-migration-workers}"
export TT_LOG_LEVEL="${TT_LOG_LEVEL:-debug}"

# Validate binaries exist
for bin in "${CONSUMER_BIN}" "${PRODUCER_BIN}"; do
  if [[ ! -x "${bin}" ]]; then
    echo "ERROR: ${bin} not found or not executable." >&2
    echo "Build with: ./build.sh --kafka" >&2
    exit 2
  fi
done

PIDS=()
LOG_DIR="./migration_test_logs"
mkdir -p "${LOG_DIR}"

cleanup() {
  echo ""
  echo "=== Shutting down ==="
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  # Give them a moment to flush
  sleep 1
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
  echo "All processes stopped."
}
trap cleanup EXIT INT TERM

echo "================================================="
echo " Migration Workers E2E Test"
echo "================================================="
echo " Kafka Brokers:  ${KAFKA_BROKERS}"
echo " Topic:          ${KAFKA_OFFLOAD_TOPIC_NAME}"
echo " Group ID:       ${KAFKA_GROUP_ID}"
echo " Workers:        ${NUM_WORKERS}"
echo " Producer msgs:  ${PRODUCER_COUNT}"
echo " Interval:       ${PRODUCER_INTERVAL_MS}ms"
echo " Log dir:        ${LOG_DIR}"
echo "================================================="
echo ""

# Start workers
for i in $(seq 1 "${NUM_WORKERS}"); do
  port=$((WORKER_BASE_PORT + i))
  log_file="${LOG_DIR}/worker_${i}.log"
  echo "Starting worker ${i} on port ${port} (log: ${log_file})"
  "${CONSUMER_BIN}" -p "${port}" > "${log_file}" 2>&1 &
  PIDS+=($!)
done

# Give workers time to join the consumer group and rebalance
echo ""
echo "Waiting 10s for consumer group rebalance..."
sleep 10

# Start producer
echo ""
echo "Starting producer (${PRODUCER_COUNT} messages, ${PRODUCER_INTERVAL_MS}ms interval)..."
producer_log="${LOG_DIR}/producer.log"
"${PRODUCER_BIN}" \
  --count "${PRODUCER_COUNT}" \
  --interval-ms "${PRODUCER_INTERVAL_MS}" \
  > "${producer_log}" 2>&1 &
PIDS+=($!)
producer_pid=$!

# Wait for producer to finish
echo "Waiting for producer (PID ${producer_pid}) to finish..."
wait "${producer_pid}" || true
echo "Producer done."

# Give workers time to process all messages
echo "Waiting 5s for workers to drain..."
sleep 5

echo ""
echo "================================================="
echo " Results"
echo "================================================="
echo ""

# Count processed messages per worker
total_processed=0
for i in $(seq 1 "${NUM_WORKERS}"); do
  log_file="${LOG_DIR}/worker_${i}.log"
  count=$(grep -c "OFFLOAD REQUEST RECEIVED" "${log_file}" 2>/dev/null || true)
  count="${count:-0}"
  # Ensure count is a plain integer
  count=$(echo "${count}" | tr -d '[:space:]')
  [[ -z "${count}" || ! "${count}" =~ ^[0-9]+$ ]] && count=0
  echo "  Worker ${i}: processed ${count} messages"
  total_processed=$((total_processed + count))
done

echo ""
echo "  Total processed: ${total_processed} / ${PRODUCER_COUNT} expected"
echo ""

if [[ ${total_processed} -ge ${PRODUCER_COUNT} ]]; then
  echo "RESULT: PASS ✅"
  exit 0
elif [[ ${total_processed} -gt 0 ]]; then
  echo "RESULT: PARTIAL (some messages received) ⚠️"
  exit 0
else
  echo "RESULT: FAIL ❌ (no messages received)"
  exit 1
fi
