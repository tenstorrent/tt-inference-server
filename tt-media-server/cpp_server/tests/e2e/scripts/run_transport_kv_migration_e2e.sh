#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Single-host run of the real KV-cache migration harness: launches a decode
# "receiver" and a prefill "sender" as two processes on 127.0.0.1 and reports
# the verification result. Requires a Mooncake build (./build.sh --mooncake),
# which produces build/transport_kv_migration_e2e.
#
# NOTE: the default host + builtin-table path is also covered by the gtest
# `TransportKvMigrationE2E` (run via ctest / the binary with no args) — it
# fork()s the sender itself and asserts byte content. This script is the way to
# drive the paths that gtest can't: MODE=device (hardware) and TABLE=<pb>
# (real protobuf tables, -DENABLE_KV_TABLE=ON).
#
# Unlike run_transport_migration_e2e.sh (the dummy single-tensor PoC), this
# drives the real table-addressed RDMA-over-host path: the double-pinned bounce
# bounce buffer, device-group fan-out, and the windowed BeginMigration/BounceReady/
# WindowReady/WindowAck/DoneMarker/Ack control flow.
#
# Usage:
#   tests/e2e/scripts/run_transport_kv_migration_e2e.sh [path-to-binary]
#
# Env overrides:
#   ROLE=both|receiver|sender  (default both — both procs on this host/loopback;
#                          receiver/sender launch ONE side for a two-galaxy run)
#   MODE=host|device      (default host — runs anywhere; device is DRISC NOC-DMA
#                          and needs HW + --blaze + MIGRATION_DRISC_SERVICE_ELF)
#   PROTOCOL=tcp|rdma      (Mooncake transport; default tcp. rdma needs an RDMA
#                          NIC + an RDMA-enabled Mooncake build)
#   TRANSPORT=bounce         (RDMA-over-host bounce buffer; the only path)
#   METADATA=<uri>         (Mooncake segment metadata; default P2PHANDSHAKE =
#                          direct peer connect. A service URI e.g.
#                          http://HOST:PORT/metadata resolves segments via that
#                          service. Applies to both roles; control stays a
#                          direct dial)
#   BOUNCE_SECTIONS=N BOUNCE_SECTION_SIZE=B  (bounce receiver geometry; 0/unset = default.
#                          a small bounce buffer — e.g. BOUNCE_SECTIONS=1 — forces the
#                          multi-window credit handshake over the wire)
#   TABLE=builtin|<path>   (default builtin; <path> needs -DENABLE_KV_TABLE=ON)
#   DECODE_TABLE=<path>    (sender, real-table mode: the decode-side .pb)
#   CONTROL_HOST=<ip>      (sender: where the receiver's control channel lives;
#                          default 127.0.0.1 — set to the decode host's IP for 2 galaxies)
#   CONTROL_PORT=N         (default 18650; the TCP control channel port)
#   RECV_NAME/SEND_NAME    (Mooncake local_server_name; default 127.0.0.1:17777/17778
#                          — set to each host's ROUTABLE IP for a two-galaxy run)
#   SLOT, LAYER_BEGIN, LAYER_END, POS_BEGIN, POS_END   (migration request;
#                          must MATCH on both sides in a two-galaxy run)
#   TIMEOUT_SEC            (default 60; per-op socket timeout on the control
#                          channel — bounds accept()/read()/write() so a hung
#                          peer can't wedge the run forever)
#
# Examples:
#   # No hardware — full orchestration + Mooncake TCP, host-backed device store:
#   tests/e2e/scripts/run_transport_kv_migration_e2e.sh
#
#   # Real device DRAM (built-in reduced table) on hardware (single host loopback):
#   MODE=device tests/e2e/scripts/run_transport_kv_migration_e2e.sh
#
#   # Two galaxies (PREFILL_IP=sender host, DECODE_IP=receiver host):
#   #   on the decode host, FIRST:
#   ROLE=receiver MODE=device RECV_NAME=${DECODE_IP}:17777 \
#     tests/e2e/scripts/run_transport_kv_migration_e2e.sh
#   #   on the prefill host:
#   ROLE=sender   MODE=device SEND_NAME=${PREFILL_IP}:17778 CONTROL_HOST=${DECODE_IP} \
#     tests/e2e/scripts/run_transport_kv_migration_e2e.sh
#   # Real tables: add TABLE=<prefill.pb> DECODE_TABLE=<decode.pb> on the sender,
#   #              and TABLE=<decode.pb> on the receiver (needs -DENABLE_KV_TABLE=ON).
set -uo pipefail

BIN="${1:-./build/transport_kv_migration_e2e}"
ROLE="${ROLE:-both}"
MODE="${MODE:-host}"
PROTOCOL="${PROTOCOL:-tcp}"
TRANSPORT="${TRANSPORT:-bounce}"
METADATA="${METADATA:-P2PHANDSHAKE}"
BOUNCE_SECTIONS="${BOUNCE_SECTIONS:-0}"
BOUNCE_SECTION_SIZE="${BOUNCE_SECTION_SIZE:-0}"
TABLE="${TABLE:-builtin}"
DECODE_TABLE="${DECODE_TABLE:-}"
CONTROL_HOST="${CONTROL_HOST:-127.0.0.1}"
CONTROL_PORT="${CONTROL_PORT:-18650}"
RECV_NAME="${RECV_NAME:-127.0.0.1:17777}"
SEND_NAME="${SEND_NAME:-127.0.0.1:17778}"
SLOT="${SLOT:-5}"
LAYER_BEGIN="${LAYER_BEGIN:-0}"; LAYER_END="${LAYER_END:-2}"
POS_BEGIN="${POS_BEGIN:-0}";     POS_END="${POS_END:-128}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
# Seed a dummy blob at the source + byte-verify the destination. Auto for
# builtin; needed to verify a real table with no model loaded. Set 0 to skip
# (real-table run against a live model, where the device already holds real KV).
SEED_VERIFY="${SEED_VERIFY:-1}"
# Device mode item-1 chip map ('mesh chip umd_chip_id' per line); optional.
DEVICE_MAP="${DEVICE_MAP:-}"
# Host tags in the tables. For real .pb these MUST match the tables'
# fabric_node_host (else the request resolves to no chunks). builtin defaults
# to prefill/decode. Leave empty to use the binary's defaults.
PREFILL_HOST="${PREFILL_HOST:-}"
DECODE_HOST="${DECODE_HOST:-}"

case "${ROLE}" in both|receiver|sender) ;; *)
  echo "ERROR: ROLE must be both|receiver|sender (got '${ROLE}')" >&2; exit 2 ;;
esac

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: ${BIN} not found/executable. Build with: ./build.sh --mooncake" >&2
  exit 2
fi

if [[ "${MODE}" == "device" && -z "${TT_METAL_RUNTIME_ROOT:-}" && -n "${TT_METAL_HOME:-}" ]]; then
  export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
fi

req=(--slot "${SLOT}" --layer-begin "${LAYER_BEGIN}" --layer-end "${LAYER_END}"
     --pos-begin "${POS_BEGIN}" --pos-end "${POS_END}"
     --mode "${MODE}"
     --protocol "${PROTOCOL}" --transport "${TRANSPORT}" --table "${TABLE}"
     --metadata "${METADATA}"
     --control-port "${CONTROL_PORT}" --timeout-sec "${TIMEOUT_SEC}")
[[ "${BOUNCE_SECTIONS}" != "0" ]] && req+=(--bounce-sections "${BOUNCE_SECTIONS}")
[[ "${BOUNCE_SECTION_SIZE}" != "0" ]] && req+=(--bounce-section-size "${BOUNCE_SECTION_SIZE}")
[[ "${SEED_VERIFY}" == "1" ]] && req+=(--seed-verify)
[[ -n "${DEVICE_MAP}" ]] && req+=(--device-map "${DEVICE_MAP}")
[[ -n "${PREFILL_HOST}" ]] && req+=(--prefill-host "${PREFILL_HOST}")
[[ -n "${DECODE_HOST}" ]] && req+=(--decode-host "${DECODE_HOST}")

send_extra=()
[[ -n "${DECODE_TABLE}" ]] && send_extra=(--decode-table "${DECODE_TABLE}")

run_receiver() {  # foreground; returns its exit code
  echo "Launching receiver (control :${CONTROL_PORT}, Mooncake ${RECV_NAME}, mode=${MODE}, table=${TABLE})..."
  "${BIN}" --role receiver --mooncake-name "${RECV_NAME}" "${req[@]}"
}

run_sender() {  # foreground; returns its exit code
  echo "Launching sender (Mooncake ${SEND_NAME} -> control ${CONTROL_HOST}:${CONTROL_PORT})..."
  "${BIN}" --role sender --mooncake-name "${SEND_NAME}" \
    --control-host "${CONTROL_HOST}" "${req[@]}" "${send_extra[@]}"
}

# Two-galaxy run: launch exactly one side on this host. The receiver must be
# started first (it binds the control port + advertises its Mooncake segment);
# then start the sender on the prefill host with CONTROL_HOST=<this host>.
if [[ "${ROLE}" == "receiver" ]]; then
  run_receiver; rc=$?
  echo "----------------------------------------"
  echo "receiver exit=${rc}"
  [[ ${rc} -eq 0 ]] && { echo "RESULT: PASS"; exit 0; } || { echo "RESULT: FAIL"; exit 1; }
fi

if [[ "${ROLE}" == "sender" ]]; then
  run_sender; rc=$?
  echo "----------------------------------------"
  echo "sender exit=${rc}"
  [[ ${rc} -eq 0 ]] && { echo "RESULT: PASS"; exit 0; } || { echo "RESULT: FAIL"; exit 1; }
fi

# ROLE=both (default): single-host loopback — receiver in the background, sender
# in the foreground. CONTROL_HOST defaults to 127.0.0.1.
run_receiver &
recv_pid=$!

# Give the receiver time to init Mooncake and bind the control port.
sleep 1

run_sender
send_rc=$?

wait "${recv_pid}"; recv_rc=$?

echo "----------------------------------------"
echo "sender exit=${send_rc}  receiver exit=${recv_rc}"
if [[ ${send_rc} -eq 0 && ${recv_rc} -eq 0 ]]; then
  echo "RESULT: PASS"; exit 0
fi
echo "RESULT: FAIL"; exit 1
