#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# local_kv_migration_lab.sh — single-host mesh of mooncake_kv_migration_worker
# (production binary) for control-plane TABLE_EXCHANGE bring-up.
#
# Unlike the multi-host deploy (one worker per box, shared CONTROL_PORT), this
# lab co-locates N prefills + M decodes on 127.0.0.1 with per-worker health +
# control ports.
#
# Usage:
#   ./local_kv_migration_lab.sh up [--prefill N] [--decode N]
#   ./local_kv_migration_lab.sh status
#   ./local_kv_migration_lab.sh logs [NAME]
#   ./local_kv_migration_lab.sh down
#
# Env:
#   WORKER_BIN, PREFILL_TABLE, DECODE_TABLE, TABLE_HOST (fabric_node_host in .pb),
#   HEALTH_PORT_BASE (19100), CONTROL_PORT_BASE (18650), META_PORT (18082),
#   STATE_DIR (/tmp/tt_mc_kv_lab), MC_TCP_BIND_ADDRESS (127.0.0.1),
#   KAFKA_BROKERS (localhost:9092 — rdkafka tolerates a missing broker for READY),
#   WORKER_PEERS_<tag>  e.g. WORKER_PEERS_prefill_0=decode-1
#     (bash assoc arrays can't be exported; use underscore form, tag '-' -> '_')
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly META_SERVER="${SCRIPT_DIR}/../../integration/run_mooncake_metadata_server.sh"
readonly STATE_DIR="${STATE_DIR:-/tmp/tt_mc_kv_lab}"
readonly BIND="127.0.0.1"

NUM_PREFILL="${NUM_PREFILL:-1}"
NUM_DECODE="${NUM_DECODE:-1}"
WORKER_BIN="${WORKER_BIN:-${CPP_ROOT}/build/mooncake_kv_migration_worker}"
PREFILL_TABLE="${PREFILL_TABLE:-/tmp/lab_kv_table.pb}"
DECODE_TABLE="${DECODE_TABLE:-${PREFILL_TABLE}}"
TABLE_HOST="${TABLE_HOST:-lab}"
HEALTH_PORT_BASE="${HEALTH_PORT_BASE:-19100}"
CONTROL_PORT_BASE="${CONTROL_PORT_BASE:-18650}"
META_PORT="${META_PORT:-18082}"
MC_TCP_BIND_ADDRESS="${MC_TCP_BIND_ADDRESS:-${BIND}}"
KAFKA_BROKERS="${KAFKA_BROKERS:-localhost:9092}"
READY_WAIT_SEC="${READY_WAIT_SEC:-90}"

die() { echo "ERROR: $*" >&2; exit 2; }
metaUri() { echo "http://${BIND}:${META_PORT}/metadata"; }

rankOf() {
  local n="$1"
  case "$n" in
    prefill-*) echo "${n#prefill-}" ;;
    decode-*) echo $(( NUM_PREFILL + ${n#decode-} )) ;;
    *) die "bad worker name '$n'" ;;
  esac
}
healthPortOf() { echo $(( HEALTH_PORT_BASE + $(rankOf "$1") )); }
controlPortOf() {
  local n="$1"
  case "$n" in
    decode-*) echo $(( CONTROL_PORT_BASE + ${n#decode-} )) ;;
    *) echo 0 ;;
  esac
}

allNames() {
  local i
  for (( i = 0; i < NUM_PREFILL; i++ )); do echo "prefill-${i}"; done
  for (( i = 0; i < NUM_DECODE; i++ )); do echo "decode-${i}"; done
}

# Default peers: prefill → every decode; decode → none.
# Override via WORKER_PEERS_<tag_with_underscores>, e.g. WORKER_PEERS_prefill_0=decode-1
peersFor() {
  # Split locals: with set -u, `local a="$1" b="${a...}"` sees a as unset.
  local workerName="$1"
  local role="${workerName%%-*}"
  local envKey="WORKER_PEERS_${workerName//-/_}"
  local override=""
  # set -u safe indirect lookup (${!var} faults when the named env is unset).
  if [[ -v "${envKey}" ]]; then
    override="${!envKey}"
  fi
  if [[ -n "${override}" ]]; then
    printf '%s' "${override}"
    return
  fi
  if [[ "${role}" == "prefill" ]]; then
    local i csv=""
    for (( i = 0; i < NUM_DECODE; i++ )); do
      csv="${csv:+${csv},}decode-${i}"
    done
    printf '%s' "${csv}"
  fi
}

saveConfig() {
  mkdir -p "${STATE_DIR}"
  cat >"${STATE_DIR}/config" <<EOF
NUM_PREFILL=${NUM_PREFILL}
NUM_DECODE=${NUM_DECODE}
WORKER_BIN=${WORKER_BIN}
PREFILL_TABLE=${PREFILL_TABLE}
DECODE_TABLE=${DECODE_TABLE}
TABLE_HOST=${TABLE_HOST}
HEALTH_PORT_BASE=${HEALTH_PORT_BASE}
CONTROL_PORT_BASE=${CONTROL_PORT_BASE}
META_PORT=${META_PORT}
MC_TCP_BIND_ADDRESS=${MC_TCP_BIND_ADDRESS}
KAFKA_BROKERS=${KAFKA_BROKERS}
EOF
}
loadConfig() { [[ -f "${STATE_DIR}/config" ]] && source "${STATE_DIR}/config"; }

clearRpcMeta() {
  curl -sS -X DELETE "$(metaUri)?key=mooncake/rpc_meta/$1" >/dev/null 2>&1 || true
  curl -sS -X DELETE "$(metaUri)?key=kv_control/$1" >/dev/null 2>&1 || true
}

launchOne() {
  # Split locals: with set -u, `local a="$1" b="${a...}"` sees a as unset.
  local workerName="$1"
  local role="${workerName%%-*}"
  local health control peers log
  local -a args=()
  health="$(healthPortOf "${workerName}")"
  control="$(controlPortOf "${workerName}")"
  peers="$(peersFor "${workerName}")"
  log="${STATE_DIR}/${workerName}.log"
  clearRpcMeta "${workerName}"

  args=(
    --role "${role}"
    --metadata "$(metaUri)"
    --name "${workerName}"
    --host "${TABLE_HOST}"
    --health-host "${BIND}"
    --health-port "${health}"
  )

  if [[ "${role}" == "decode" ]]; then
    args+=(--table "${DECODE_TABLE}" --control-port "${control}")
  else
    [[ -n "${peers}" ]] || die "prefill '${workerName}' has empty peers"
    args+=(--prefill-table "${PREFILL_TABLE}" --decode-table "${DECODE_TABLE}")
    local tag
    IFS=',' read -ra peerTags <<<"${peers}"
    for tag in "${peerTags[@]}"; do
      [[ -n "${tag}" ]] && args+=(--peer "${tag}")
    done
  fi

  echo "[lab] ${workerName} peers='${peers}' health=${health} control=${control}"
  MC_TCP_BIND_ADDRESS="${MC_TCP_BIND_ADDRESS}" \
  KAFKA_BROKERS="${KAFKA_BROKERS}" \
  KAFKA_GROUP_ID="migration-workers-prefill-${workerName}" \
    "${WORKER_BIN}" "${args[@]}" >"${log}" 2>&1 &
  echo $! >"${STATE_DIR}/${workerName}.pid"
}

httpCode() {
  curl -so /dev/null -w '%{http_code}' --max-time 2 "http://${BIND}:$1$2" 2>/dev/null || echo "---"
}

waitReady() {
  local deadline=$(( SECONDS + READY_WAIT_SEC )) name port code
  echo "[lab] waiting up to ${READY_WAIT_SEC}s for /readyz on all workers..."
  while (( SECONDS < deadline )); do
    local allOk=1
    for name in $(allNames); do
      port="$(healthPortOf "${name}")"
      code="$(httpCode "${port}" /readyz)"
      if [[ "${code}" != "200" ]]; then
        allOk=0
        break
      fi
    done
    if (( allOk )); then
      echo "[lab] all workers ready"
      return 0
    fi
    sleep 1
  done
  echo "[lab] WARN: not all workers ready within ${READY_WAIT_SEC}s" >&2
  cmdStatus
  return 1
}

cmdUp() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --prefill) NUM_PREFILL="$2"; shift 2 ;;
      --decode) NUM_DECODE="$2"; shift 2 ;;
      *) die "unknown up arg: $1" ;;
    esac
  done
  [[ -x "${WORKER_BIN}" ]] || die "${WORKER_BIN} not executable"
  [[ -f "${PREFILL_TABLE}" ]] || die "PREFILL_TABLE missing: ${PREFILL_TABLE}"
  [[ -f "${DECODE_TABLE}" ]] || die "DECODE_TABLE missing: ${DECODE_TABLE}"
  if [[ -f "${STATE_DIR}/metadata.pid" ]] && kill -0 "$(cat "${STATE_DIR}/metadata.pid")" 2>/dev/null; then
    die "lab already up (${STATE_DIR}); run '$0 down' first"
  fi
  mkdir -p "${STATE_DIR}"
  saveConfig
  echo "[lab] metadata on ${BIND}:${META_PORT}"
  HTTP_PORT="${META_PORT}" BIND_HOST="${BIND}" \
  PYTHON="${PYTHON:-$(command -v python3)}" \
    bash "${META_SERVER}" >"${STATE_DIR}/metadata.log" 2>&1 &
  echo $! >"${STATE_DIR}/metadata.pid"
  sleep 2

  # Decodes first so control endpoints are published before prefills dial.
  local n
  for (( n = 0; n < NUM_DECODE; n++ )); do launchOne "decode-${n}"; done
  sleep 2
  for (( n = 0; n < NUM_PREFILL; n++ )); do launchOne "prefill-${n}"; done
  waitReady || true
  cmdStatus
  cmdVerify
}

cmdStatus() {
  loadConfig
  printf '%-12s %-10s %-14s %-9s %-8s %-10s\n' NAME PID HEALTH HEALTHZ READYZ CONTROL
  local n pid port shown ctrl
  for n in $(allNames); do
    pid="$(cat "${STATE_DIR}/${n}.pid" 2>/dev/null || echo -)"
    port="$(healthPortOf "${n}")"
    ctrl="$(controlPortOf "${n}")"
    shown="dead"
    [[ "${pid}" != "-" ]] && kill -0 "${pid}" 2>/dev/null && shown="${pid}"
    printf '%-12s %-10s %-14s %-9s %-8s %-10s\n' \
      "${n}" "${shown}" "${BIND}:${port}" "$(httpCode "${port}" /healthz)" \
      "$(httpCode "${port}" /readyz)" "${ctrl}"
  done
}

cmdVerify() {
  loadConfig
  echo "---- TABLE_EXCHANGE / READY / peer connect ----"
  local n log
  for n in $(allNames); do
    log="${STATE_DIR}/${n}.log"
    echo "== ${n} =="
    grep -E "READY:|TABLE_EXCHANGE|got decode table via control|published control|discovered peer|CONNECTED|control channel|awaitConnected|failed" \
      "${log}" 2>/dev/null | tail -40 || echo "(no matches / missing log)"
  done
}

cmdLogs() {
  loadConfig
  local name="${1:-}"
  if [[ -n "${name}" ]]; then
    sed -n '1,200p' "${STATE_DIR}/${name}.log"
  else
    cmdVerify
  fi
}

cmdDown() {
  echo "[lab] tearing down ${STATE_DIR}"
  # Kill only via recorded PIDs — never pkill -f (matches this script's cmdline).
  if [[ -d "${STATE_DIR}" ]]; then
    local pf
    for pf in "${STATE_DIR}"/*.pid; do
      [[ -f "${pf}" ]] || continue
      kill "$(cat "${pf}")" 2>/dev/null || true
    done
  fi
  rm -rf "${STATE_DIR}"
}

case "${1:-}" in
  up) shift; cmdUp "$@" ;;
  status) cmdStatus ;;
  verify) cmdVerify ;;
  logs) shift; cmdLogs "$@" ;;
  down) cmdDown ;;
  *)
    echo "usage: $0 {up [--prefill N] [--decode N]|status|verify|logs [NAME]|down}"
    exit 1
    ;;
esac
