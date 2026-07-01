#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Multi-host / multi-galaxy (1 prefill -> N decode) run of the KV-migration
# harness. Each decode process byte-verifies its own slice; the prefill process
# fans out to all of them (KvMigrationMultiHostSender). Requires a Mooncake
# build (./build.sh --mooncake); real tables need --kv-table; Kafka trigger
# needs --kafka + a broker.
#
# THREE ROLES (set ROLE=...):
#
#   both      (default) — everything on ONE box (loopback). Launches N decode
#             receivers + 1 sender locally. Use PEERS="tag:cport:mport ..." and
#             HOST_IP (default 127.0.0.1). Good for host-mode fan-out tests.
#
#   receiver  — run on EACH decode host/galaxy. Launches this host's ONE
#             receiver. Set: DECODE_HOST=<tag> MOONCAKE_NAME=<thisIP>:<port>
#             CONTROL_PORT=<port> TABLE=<decode.pb|builtin2> [DEVICE_MAP=...].
#             Start the receivers BEFORE the sender.
#
#   sender    — run on the prefill host/galaxy. Fans out to the decode hosts.
#             Set: SEND_NAME=<prefillIP>:<port> TABLE=<prefill.pb|builtin2>
#             DECODE_TABLE=<decode.pb> PEERS="tag=ip:cport tag=ip:cport ..."
#             [TRIGGER=kafka KAFKA_BROKERS=...].
#
# The request (SLOT/LAYER_*/POS_*) MUST match on every process. Mooncake uses
# P2PHANDSHAKE (no metadata server): MOONCAKE_NAME / SEND_NAME must be ROUTABLE
# ip:port with the port open; the decode CONTROL_PORT must be reachable from the
# prefill host. See tests/e2e/TRANSPORT_KV_MIGRATION_E2E.md for full details.
#
# Common env: MODE=host|device, SEED_VERIFY=1|0 (default 1), DEVICE_MAP=<file>,
#   SLOT/LAYER_BEGIN/LAYER_END/POS_BEGIN/POS_END, TIMEOUT_SEC.
set -uo pipefail

BIN="${1:-./build/transport_kv_migration_e2e}"
ROLE="${ROLE:-both}"
MODE="${MODE:-host}"
TABLE="${TABLE:-builtin2}"        # THIS role's local table.
DECODE_TABLE="${DECODE_TABLE:-}"  # sender real-table mode: the decode .pb.
DEVICE_MAP="${DEVICE_MAP:-}"
SEED_VERIFY="${SEED_VERIFY:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
TRIGGER="${TRIGGER:-cli}"
KAFKA_BROKERS="${KAFKA_BROKERS:-localhost:9092}"
SLOT="${SLOT:-5}"
LAYER_BEGIN="${LAYER_BEGIN:-0}"; LAYER_END="${LAYER_END:-2}"
POS_BEGIN="${POS_BEGIN:-0}";     POS_END="${POS_END:-128}"

# Peer list. Format depends on role: both => "tag:cport:mport" (shared HOST_IP);
# sender => "tag=ip:cport". No shared default (a both-format default would be
# wrong for the sender role); both-mode fills its own default below.
PEERS="${PEERS:-}"
HOST_IP="${HOST_IP:-127.0.0.1}"
SEND_NAME="${SEND_NAME:-127.0.0.1:17780}"
# receiver-mode (this host):
DECODE_HOST="${DECODE_HOST:-decode-0}"
CONTROL_PORT="${CONTROL_PORT:-18650}"
MOONCAKE_NAME="${MOONCAKE_NAME:-127.0.0.1:17777}"

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

# Shared request/mode flags (NOT --table: that's per role).
common=(--mode "${MODE}" --slot "${SLOT}"
        --layer-begin "${LAYER_BEGIN}" --layer-end "${LAYER_END}"
        --pos-begin "${POS_BEGIN}" --pos-end "${POS_END}"
        --timeout-sec "${TIMEOUT_SEC}")
[[ "${SEED_VERIFY}" == "1" ]] && common+=(--seed-verify)
[[ -n "${DEVICE_MAP}" ]] && common+=(--device-map "${DEVICE_MAP}")
# Prefill host tag (sender only; must match the prefill .pb's fabric_node_host
# for real tables). Receiver's tag is DECODE_HOST / the PEERS tags.
PREFILL_HOST="${PREFILL_HOST:-}"
prefill_flag=(); [[ -n "${PREFILL_HOST}" ]] && prefill_flag=(--prefill-host "${PREFILL_HOST}")

result() {  # $1 = rc
  echo "----------------------------------------"
  if [[ ${1} -eq 0 ]]; then echo "RESULT: PASS"; exit 0; fi
  echo "RESULT: FAIL"; exit 1
}

# --- ROLE=receiver: this host's single decode receiver -----------------------
if [[ "${ROLE}" == "receiver" ]]; then
  echo "Receiver '${DECODE_HOST}' (control :${CONTROL_PORT}, Mooncake ${MOONCAKE_NAME}, table=${TABLE}, mode=${MODE})..."
  "${BIN}" --role receiver --decode-host "${DECODE_HOST}" \
    --control-port "${CONTROL_PORT}" --mooncake-name "${MOONCAKE_NAME}" \
    --table "${TABLE}" "${common[@]}"
  result $?
fi

# --- ROLE=sender: fan out to the decode hosts in PEERS -----------------------
if [[ "${ROLE}" == "sender" ]]; then
  peer_flags=(); n=0
  for spec in ${PEERS}; do
    if [[ "${spec}" != *"="* ]]; then
      echo "ERROR: ROLE=sender PEERS entries must be 'tag=ip:cport' (got '${spec}')" >&2
      exit 2
    fi
    peer_flags+=(--peer-control "${spec}"); n=$((n + 1))
  done
  if (( n == 0 )); then echo "ERROR: ROLE=sender needs PEERS=\"tag=ip:cport ...\"" >&2; exit 2; fi
  send_extra=()
  [[ -n "${DECODE_TABLE}" ]] && send_extra+=(--decode-table "${DECODE_TABLE}")
  [[ "${TRIGGER}" == "kafka" ]] && send_extra+=(--trigger kafka --kafka-brokers "${KAFKA_BROKERS}")
  echo "Sender -> ${n} decode host(s) (table=${TABLE}, trigger=${TRIGGER})..."
  "${BIN}" --role sender --mooncake-name "${SEND_NAME}" --table "${TABLE}" \
    "${common[@]}" "${prefill_flag[@]}" "${peer_flags[@]}" "${send_extra[@]}"
  result $?
fi

# --- ROLE=both: everything on one box (loopback) -----------------------------
# Fill the both-mode default peer list (tag:cport:mport) if none was given.
[[ -z "${PEERS}" ]] && PEERS="decode-0:18651:17781 decode-1:18652:17782"
# Receiver decode table: builtin2 -> the split builtin; real -> the decode .pb.
if [[ "${TABLE}" == "builtin2" ]]; then
  recv_table="builtin2"
else
  recv_table="${DECODE_TABLE}"
  [[ -z "${recv_table}" ]] && { echo "ERROR: real-table both-mode needs DECODE_TABLE=<decode.pb>" >&2; exit 2; }
fi

cleanup() { for pid in "${recv_pids[@]:-}"; do kill "${pid}" 2>/dev/null; done; }
trap cleanup EXIT

recv_pids=(); peer_flags=(); n=0
for spec in ${PEERS}; do   # both-mode format: tag:cport:mport
  tag="${spec%%:*}"; rest="${spec#*:}"; cport="${rest%%:*}"; mport="${rest##*:}"
  echo "Launching receiver '${tag}' (control :${cport}, Mooncake ${HOST_IP}:${mport}, table=${recv_table})..."
  "${BIN}" --role receiver --decode-host "${tag}" --control-port "${cport}" \
    --mooncake-name "${HOST_IP}:${mport}" --table "${recv_table}" "${common[@]}" &
  recv_pids+=($!)
  peer_flags+=(--peer-control "${tag}=${HOST_IP}:${cport}")
  n=$((n + 1))
done
sleep 1
send_extra=()
[[ "${TABLE}" != "builtin2" && -n "${DECODE_TABLE}" ]] && send_extra=(--decode-table "${DECODE_TABLE}")
[[ "${TRIGGER}" == "kafka" ]] && send_extra+=(--trigger kafka --kafka-brokers "${KAFKA_BROKERS}")
echo "Launching sender -> ${n} decode host(s) (table=${TABLE}, trigger=${TRIGGER})..."
"${BIN}" --role sender --mooncake-name "${SEND_NAME}" --table "${TABLE}" \
  "${common[@]}" "${prefill_flag[@]}" "${peer_flags[@]}" "${send_extra[@]}"
send_rc=$?

all_ok=1
for pid in "${recv_pids[@]}"; do wait "${pid}" || all_ok=0; done
recv_pids=()

echo "----------------------------------------"
echo "sender exit=${send_rc}  receivers_ok=${all_ok}  (${n} decode hosts)"
[[ ${send_rc} -eq 0 && ${all_ok} -eq 1 ]] && { echo "RESULT: PASS"; exit 0; }
echo "RESULT: FAIL"; exit 1
