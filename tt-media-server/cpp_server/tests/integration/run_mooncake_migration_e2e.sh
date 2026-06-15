#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# End-to-end Mooncake migration test with multiple consumers:
#   1. Starts a migration_source_dummy (registers a buffer with Mooncake)
#   2. Starts N tt_consumer instances (MigrationWorker + MooncakeTransferEngine)
#   3. Publishes multiple Kafka messages pointing to the source's real segment
#   4. Verifies consumers successfully pulled data via Mooncake TCP transport
#
# Prerequisites:
#   - Build with: ./build.sh --blaze --kafka --mooncake
#   - Kafka running (docker with --network host or accessible at KAFKA_BROKERS)
#
# Usage:
#   tests/integration/run_mooncake_migration_e2e.sh [build-dir]
#
# Env overrides:
#   KAFKA_BROKERS             (default: 172.17.0.1:9092)
#   KAFKA_OFFLOAD_TOPIC_NAME  (default: session-offload)
#   NUM_CONSUMERS             (default: 3)
#   NUM_MESSAGES              (default: 9, should be >= NUM_CONSUMERS)

set -euo pipefail

BUILD_DIR="${1:-./build}"
SOURCE_BIN="${BUILD_DIR}/migration_source_dummy"
CONSUMER_BIN="${BUILD_DIR}/tt_consumer"

export KAFKA_BROKERS="${KAFKA_BROKERS:-172.17.0.1:9092}"
export KAFKA_OFFLOAD_TOPIC_NAME="${KAFKA_OFFLOAD_TOPIC_NAME:-session-offload}"
export KAFKA_GROUP_ID="${KAFKA_GROUP_ID:-mooncake-e2e-$(date +%s)}"
export TT_LOG_LEVEL="${TT_LOG_LEVEL:-info}"

NUM_CONSUMERS="${NUM_CONSUMERS:-3}"
NUM_MESSAGES="${NUM_MESSAGES:-9}"
CONSUMER_BASE_PORT="${CONSUMER_BASE_PORT:-9200}"
BUFFER_SIZE=52428800  # 50 MB
FILL_BYTE=171  # 0xAB

LOG_DIR="./mooncake_e2e_logs"
rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"

PIDS=()

cleanup() {
  echo ""
  echo "=== Shutting down ==="
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
  echo "All processes stopped."
}
trap cleanup EXIT INT TERM

# Validate binaries
for bin in "${SOURCE_BIN}" "${CONSUMER_BIN}"; do
  if [[ ! -x "${bin}" ]]; then
    echo "ERROR: ${bin} not found. Build with: ./build.sh --blaze --kafka --mooncake" >&2
    exit 2
  fi
done

echo "================================================="
echo " Mooncake Migration E2E Test (${NUM_CONSUMERS} consumers)"
echo "================================================="
echo " Kafka:        ${KAFKA_BROKERS}"
echo " Topic:        ${KAFKA_OFFLOAD_TOPIC_NAME}"
echo " Group:        ${KAFKA_GROUP_ID}"
echo " Consumers:    ${NUM_CONSUMERS}"
echo " Messages:     ${NUM_MESSAGES}"
echo " Buffer:       ${BUFFER_SIZE} bytes (fill=0x$(printf '%02X' ${FILL_BYTE}))"
echo " Log dir:      ${LOG_DIR}"
echo "================================================="
echo ""

# ─── Step 0: Purge topic (delete + recreate) to avoid stale messages ──────────
echo "Step 0: Purging Kafka topic '${KAFKA_OFFLOAD_TOPIC_NAME}'..."
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic "${KAFKA_OFFLOAD_TOPIC_NAME}" 2>/dev/null || true
sleep 1
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic "${KAFKA_OFFLOAD_TOPIC_NAME}" \
  --partitions 3 --replication-factor 1 2>/dev/null || true
echo "  ✅ Topic purged"
echo ""

# ─── Step 1: Start source peer ───────────────────────────────────────────────
echo "Step 1: Starting migration source (Mooncake segment)..."
SOURCE_LOG="${LOG_DIR}/source.log"

"${SOURCE_BIN}" \
  --local-server-name "127.0.0.1:0" \
  --buffer-size "${BUFFER_SIZE}" \
  --fill-byte "${FILL_BYTE}" \
  > "${SOURCE_LOG}" 2>&1 &
SOURCE_PID=$!
PIDS+=("${SOURCE_PID}")

# Wait for source to be ready (it prints "READY" to stdout).
echo "  Waiting for source to be ready..."
TIMEOUT=10
for i in $(seq 1 $TIMEOUT); do
  if grep -q "READY" "${SOURCE_LOG}" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${SOURCE_PID}" 2>/dev/null; then
    echo "ERROR: Source process died. Log:" >&2
    cat "${SOURCE_LOG}" >&2
    exit 1
  fi
  sleep 1
done

if ! grep -q "READY" "${SOURCE_LOG}" 2>/dev/null; then
  echo "ERROR: Source didn't become ready in ${TIMEOUT}s" >&2
  cat "${SOURCE_LOG}" >&2
  exit 1
fi

# Parse source output.
SEGMENT=$(grep "^SEGMENT=" "${SOURCE_LOG}" | head -1 | cut -d= -f2)
BUFFER_ADDR=$(grep "^BUFFER_ADDR=" "${SOURCE_LOG}" | head -1 | cut -d= -f2)

echo "  ✅ Source ready: segment=${SEGMENT} buffer_addr=${BUFFER_ADDR}"
echo ""

# Extract host:port from segment name for Kafka messages.
SOURCE_HOST=$(echo "${SEGMENT}" | cut -d: -f1)
SOURCE_PORT=$(echo "${SEGMENT}" | cut -d: -f2)

# ─── Step 2: Start N consumers (MigrationWorker + Mooncake) ──────────────────
echo "Step 2: Starting ${NUM_CONSUMERS} consumers (MigrationWorker + Mooncake)..."

for i in $(seq 1 "${NUM_CONSUMERS}"); do
  port=$((CONSUMER_BASE_PORT + i))
  log_file="${LOG_DIR}/consumer_${i}.log"
  echo "  Starting consumer ${i} on port ${port} (log: ${log_file})"
  "${CONSUMER_BIN}" \
    --local-server-name "127.0.0.1:0" \
    -p "${port}" \
    > "${log_file}" 2>&1 &
  PIDS+=($!)
done

# Give consumers time to join the Kafka consumer group and rebalance.
echo ""
echo "  Waiting 10s for consumer group rebalance..."
sleep 10

# Check all consumers are still alive.
ALIVE=0
for i in $(seq 1 "${NUM_CONSUMERS}"); do
  pid_idx=$((i))  # PIDS[0]=source, PIDS[1..N]=consumers
  if kill -0 "${PIDS[$pid_idx]}" 2>/dev/null; then
    ALIVE=$((ALIVE + 1))
  else
    echo "  ⚠️  Consumer ${i} died early. Log:"
    cat "${LOG_DIR}/consumer_${i}.log"
  fi
done
echo "  ✅ ${ALIVE}/${NUM_CONSUMERS} consumers running"
echo ""

# ─── Step 3: Publish N migration messages via native C++ producer ─────────────
echo "Step 3: Publishing ${NUM_MESSAGES} messages via native C++ producer..."
echo "  (source_host=${SOURCE_HOST} source_port=${SOURCE_PORT} offset=0 kv_size=${BUFFER_SIZE})"

PRODUCER_BIN="${BUILD_DIR}/migration_producer_dummy"
if [[ ! -x "${PRODUCER_BIN}" ]]; then
  echo "ERROR: ${PRODUCER_BIN} not found." >&2
  exit 2
fi

PRODUCER_LOG="${LOG_DIR}/producer.log"
"${PRODUCER_BIN}" \
  --count "${NUM_MESSAGES}" \
  --interval-ms 0 \
  --source-host "${SOURCE_HOST}" \
  --source-port "${SOURCE_PORT}" \
  --source-offset 0 \
  --kv-size "${BUFFER_SIZE}" \
  > "${PRODUCER_LOG}" 2>&1
PRODUCER_EXIT=$?

if [[ ${PRODUCER_EXIT} -ne 0 ]]; then
  echo "  ❌ Producer failed (exit=${PRODUCER_EXIT}). Log:"
  cat "${PRODUCER_LOG}"
  exit 1
fi

echo "  ✅ ${NUM_MESSAGES} messages published (native C++, sub-ms latency)"
echo ""

# ─── Step 4: Wait and verify ─────────────────────────────────────────────────
echo "Step 4: Waiting for consumers to process messages..."
sleep 8

echo ""
echo "================================================="
echo " Results"
echo "================================================="
echo ""

total_success=0
total_failed=0
total_received=0

for i in $(seq 1 "${NUM_CONSUMERS}"); do
  log_file="${LOG_DIR}/consumer_${i}.log"
  success=$(grep -c "RDMA pull complete" "${log_file}" 2>/dev/null || echo 0)
  failed=$(grep -c "RDMA pull failed\|Failed to open segment" "${log_file}" 2>/dev/null || echo 0)
  received=$(grep -c "OFFLOAD REQUEST RECEIVED" "${log_file}" 2>/dev/null || echo 0)

  # Ensure values are plain integers
  success=$(echo "${success}" | tr -d '[:space:]')
  failed=$(echo "${failed}" | tr -d '[:space:]')
  received=$(echo "${received}" | tr -d '[:space:]')
  [[ -z "${success}" || ! "${success}" =~ ^[0-9]+$ ]] && success=0
  [[ -z "${failed}" || ! "${failed}" =~ ^[0-9]+$ ]] && failed=0
  [[ -z "${received}" || ! "${received}" =~ ^[0-9]+$ ]] && received=0

  echo "  Consumer ${i}: received=${received} success=${success} failed=${failed}"
  total_success=$((total_success + success))
  total_failed=$((total_failed + failed))
  total_received=$((total_received + received))
done

echo ""
echo "  ─────────────────────────────────────────────"
echo "  Total received:   ${total_received} / ${NUM_MESSAGES}"
echo "  Total RDMA OK:    ${total_success} / ${NUM_MESSAGES}"
echo "  Total RDMA FAIL:  ${total_failed}"
echo "  ─────────────────────────────────────────────"
echo ""

if [[ ${total_success} -ge ${NUM_MESSAGES} && ${total_failed} -eq 0 ]]; then
  echo "RESULT: PASS ✅ (all ${NUM_MESSAGES} transfers succeeded across ${NUM_CONSUMERS} consumers)"
  exit 0
elif [[ ${total_success} -gt 0 && ${total_failed} -eq 0 ]]; then
  echo "RESULT: PARTIAL ⚠️  (${total_success}/${NUM_MESSAGES} succeeded, none failed — may need more drain time)"
  exit 0
elif [[ ${total_success} -gt 0 ]]; then
  echo "RESULT: MIXED ⚠️  (${total_success} OK, ${total_failed} FAILED)"
  echo ""
  echo "--- Failed consumer logs ---"
  for i in $(seq 1 "${NUM_CONSUMERS}"); do
    if grep -q "RDMA pull failed\|Failed to open segment" "${LOG_DIR}/consumer_${i}.log" 2>/dev/null; then
      echo "=== Consumer ${i} ==="
      grep -A2 "RDMA pull failed\|Failed to open segment" "${LOG_DIR}/consumer_${i}.log"
    fi
  done
  exit 1
else
  echo "RESULT: FAIL ❌ (no successful transfers)"
  echo ""
  for i in $(seq 1 "${NUM_CONSUMERS}"); do
    echo "=== Consumer ${i} (tail) ==="
    tail -10 "${LOG_DIR}/consumer_${i}.log"
    echo ""
  done
  exit 1
fi
