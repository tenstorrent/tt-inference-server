#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Single-host run of the #3890 migration acceptance harness: launches a receiver
# and a sender as two processes on 127.0.0.1 and reports the verification result.
# Requires a Mooncake-enabled build (./build.sh --mooncake), which produces
# build/transport_migration_e2e. Device mode additionally needs TT_METAL_HOME set
# at build time and real hardware.
#
# Usage:
#   tests/e2e/scripts/run_transport_migration_e2e.sh [path-to-binary]
#
# Env overrides:
#   STORAGE=host|device   (default host — runs anywhere; use device on real HW)
#   BYTES=N               (default 65536)
#   TIMEOUT_SEC=S         (default 30)
#   RECV_NAME / SEND_NAME (default 127.0.0.1:17777 / 127.0.0.1:17778)
#
#   Boards (device mode): sender owns the src board, receiver owns the dst board.
#   DEVICE_ID=N           (default 0; sets both sides unless overridden below)
#   SRC_DEVICE_ID=N       (sender's board; default = DEVICE_ID)
#   DST_DEVICE_ID=N       (receiver's board; default = DEVICE_ID)
#
#   DRAM addresses (device mode; ignored for host mode):
#   SRC_ADDR=0xNNN        (sender NocAddr; default 0x1000000)
#   DST_ADDR=0xNNN        (receiver NocAddr; default 0x2000000)
#
# Examples:
#   # Transport-only smoke test, no hardware:
#   tests/e2e/scripts/run_transport_migration_e2e.sh
#
#   # Two boards on one host, real device DRAM, different addresses:
#   STORAGE=device SRC_DEVICE_ID=0 DST_DEVICE_ID=1 BYTES=1048576 \
#     tests/e2e/scripts/run_transport_migration_e2e.sh
set -uo pipefail

BIN="${1:-./build/transport_migration_e2e}"
STORAGE="${STORAGE:-host}"
BYTES="${BYTES:-65536}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"   # device init + cold JIT can take many seconds
RECV_NAME="${RECV_NAME:-127.0.0.1:17777}"
SEND_NAME="${SEND_NAME:-127.0.0.1:17778}"

DEVICE_ID="${DEVICE_ID:-0}"
SRC_DEVICE_ID="${SRC_DEVICE_ID:-${DEVICE_ID}}"
DST_DEVICE_ID="${DST_DEVICE_ID:-${DEVICE_ID}}"
SRC_ADDR="${SRC_ADDR:-0x1000000}"
DST_ADDR="${DST_ADDR:-0x2000000}"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: ${BIN} not found/executable. Build with: ./build.sh --mooncake" >&2
  exit 2
fi

# Device mode needs tt-metal's RUNTIME root (kernels/firmware), which is separate
# from the build-time TT_METAL_HOME. Default it from TT_METAL_HOME if unset.
if [[ "${STORAGE}" == "device" && -z "${TT_METAL_RUNTIME_ROOT:-}" ]]; then
  if [[ -n "${TT_METAL_HOME:-}" ]]; then
    export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
    echo "TT_METAL_RUNTIME_ROOT defaulted to TT_METAL_HOME=${TT_METAL_HOME}"
  else
    echo "WARNING: device mode but TT_METAL_RUNTIME_ROOT/TT_METAL_HOME unset;" \
         "CreateDevice will fail to find its root." >&2
  fi
fi

# Under P2PHANDSHAKE the receiver binds a random port, so its real segment name
# is not known ahead of time. The receiver writes it to this rendezvous file and
# the sender reads it (the --local ports below are just bind hints / formatting).
RDV="${RENDEZVOUS:-$(mktemp -u /tmp/tt_mig_rdv.XXXXXX)}"
rm -f "${RDV}" "${RDV}.tmp"
cleanup() { rm -f "${RDV}" "${RDV}.tmp"; }
trap cleanup EXIT

base=(--storage "${STORAGE}" --bytes "${BYTES}" --timeout-sec "${TIMEOUT_SEC}"
      --rendezvous "${RDV}")

# In device mode, each process must open ONLY its own board: tt-metal's
# CreateDevice opens (and exclusively locks) the whole chip cluster, so two
# processes on one host would deadlock on the per-chip lock. TT_VISIBLE_DEVICES
# (UMD) restricts the process to one PCIe device; inside it that board is logical
# id 0. In host mode these are unused.
recv_env=(); send_env=()
if [[ "${STORAGE}" == "device" ]]; then
  recv_env=(env "TT_VISIBLE_DEVICES=${DST_DEVICE_ID}")
  send_env=(env "TT_VISIBLE_DEVICES=${SRC_DEVICE_ID}")
  recv_args=("${base[@]}" --device-id 0 --device-addr "${DST_ADDR}")
  send_args=("${base[@]}" --device-id 0 --device-addr "${SRC_ADDR}")
else
  recv_args=("${base[@]}" --device-id "${DST_DEVICE_ID}" --device-addr "${DST_ADDR}")
  send_args=("${base[@]}" --device-id "${SRC_DEVICE_ID}" --device-addr "${SRC_ADDR}")
fi

echo "Launching receiver (${RECV_NAME}, board ${DST_DEVICE_ID} @ ${DST_ADDR})..."
"${recv_env[@]}" "${BIN}" --role receiver --local "${RECV_NAME}" "${recv_args[@]}" &
recv_pid=$!

# Give the receiver a moment to init the device and register its segment.
sleep 1

echo "Launching sender (${SEND_NAME} -> board ${SRC_DEVICE_ID} @ ${SRC_ADDR})..."
"${send_env[@]}" "${BIN}" --role sender --local "${SEND_NAME}" "${send_args[@]}"
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
