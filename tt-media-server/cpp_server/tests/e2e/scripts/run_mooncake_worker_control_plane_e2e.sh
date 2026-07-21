#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Control-plane bring-up e2e for mooncake_kv_migration_worker (NOT KV migrate).
#
# Gates the production glue that always runs before migrate:
#   load .pb → DeviceMap socket handoff → rpc_meta → kv_control →
#   peer connect → TABLE_EXCHANGE → READY
#
# Default mesh: 1 prefill + 2 decode on localhost. Fail-closed: any missing
# peer, failed handoff, failed TABLE_EXCHANGE, or /readyz timeout → exit 1.
#
# Distinct from MooncakeMpiDiscovery (bringup_mooncake_worker CONNECTED-only)
# and from lab/nightly migrate e2e (Kafka seed → tensor transfer → verify).
#
# Env:
#   WORKER_BIN, HANDOFF_SENDER_BIN, FIXTURE_DIR, STATE_DIR,
#   NUM_PREFILL (1), NUM_DECODE (2), READY_WAIT_SEC (120),
#   META_PORT (18083), HEALTH_PORT_BASE (19200), CONTROL_PORT_BASE (18750),
#   HANDOFF_PORT_BASE (18800), KAFKA_BROKERS (localhost:9092 — optional broker)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly META_SERVER="${SCRIPT_DIR}/../../integration/run_mooncake_metadata_server.sh"
readonly FIXTURE_GEN="${SCRIPT_DIR}/gen_control_plane_fixtures.py"
readonly BIND="127.0.0.1"
readonly TABLE_HOST="ci-host"

NUM_PREFILL="${NUM_PREFILL:-1}"
NUM_DECODE="${NUM_DECODE:-2}"
WORKER_BIN="${WORKER_BIN:-${CPP_ROOT}/build/mooncake_kv_migration_worker}"
HANDOFF_SENDER_BIN="${HANDOFF_SENDER_BIN:-${CPP_ROOT}/build/engine_handoff_sender}"
FIXTURE_DIR="${FIXTURE_DIR:-${SCRIPT_DIR}/../fixtures/control_plane}"
STATE_DIR="${STATE_DIR:-/tmp/tt_mc_control_plane_e2e}"
HEALTH_PORT_BASE="${HEALTH_PORT_BASE:-19200}"
CONTROL_PORT_BASE="${CONTROL_PORT_BASE:-18750}"
HANDOFF_PORT_BASE="${HANDOFF_PORT_BASE:-18800}"
META_PORT="${META_PORT:-18083}"
MC_TCP_BIND_ADDRESS="${MC_TCP_BIND_ADDRESS:-${BIND}}"
KAFKA_BROKERS="${KAFKA_BROKERS:-localhost:9092}"
READY_WAIT_SEC="${READY_WAIT_SEC:-120}"

die() { echo "ERROR: $*" >&2; exit 1; }
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
handoffPortOf() { echo $(( HANDOFF_PORT_BASE + $(rankOf "$1") )); }

allNames() {
  local i
  for (( i = 0; i < NUM_PREFILL; i++ )); do echo "prefill-${i}"; done
  for (( i = 0; i < NUM_DECODE; i++ )); do echo "decode-${i}"; done
}

peersFor() {
  local workerName="$1"
  local role="${workerName%%-*}"
  if [[ "${role}" != "prefill" ]]; then
    return 0
  fi
  local i csv=""
  for (( i = 0; i < NUM_DECODE; i++ )); do
    csv="${csv:+${csv},}decode-${i}"
  done
  printf '%s' "${csv}"
}

ensureFixtures() {
  local prefillPb="${FIXTURE_DIR}/prefill.pb"
  local decodePb="${FIXTURE_DIR}/decode.pb"
  local devmap="${FIXTURE_DIR}/ci-host.devmap"
  if [[ -f "${prefillPb}" && -f "${decodePb}" && -f "${devmap}" ]]; then
    return 0
  fi
  [[ -f "${FIXTURE_GEN}" ]] || die "missing fixture gen: ${FIXTURE_GEN}"
  python3 "${FIXTURE_GEN}" -o "${FIXTURE_DIR}"
  [[ -f "${prefillPb}" && -f "${decodePb}" && -f "${devmap}" ]] \
    || die "fixture generation failed under ${FIXTURE_DIR}"
}

clearRpcMeta() {
  curl -sS -X DELETE "$(metaUri)?key=mooncake/rpc_meta/$1" >/dev/null 2>&1 || true
  curl -sS -X DELETE "$(metaUri)?key=kv_control/$1" >/dev/null 2>&1 || true
}

httpCode() {
  curl -so /dev/null -w '%{http_code}' --max-time 2 "http://${BIND}:$1$2" 2>/dev/null || echo "---"
}

probeMetadata() {
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

pushDeviceMap() {
  local workerName="$1"
  local handoffPort deviceMapPath attempt
  handoffPort="$(handoffPortOf "${workerName}")"
  deviceMapPath="${FIXTURE_DIR}/ci-host.devmap"
  for (( attempt = 0; attempt < 60; attempt++ )); do
    if "${HANDOFF_SENDER_BIN}" \
         --host "${BIND}" --port "${handoffPort}" \
         --device-map "${deviceMapPath}" >/dev/null 2>&1; then
      echo "[control-plane] DeviceMap pushed to ${workerName} :${handoffPort}"
      return 0
    fi
    sleep 1
  done
  die "DeviceMap handoff to ${workerName} on :${handoffPort} failed after 60s"
}

launchOne() {
  local workerName="$1"
  local role="${workerName%%-*}"
  local health control handoff peers log
  local -a args=()
  health="$(healthPortOf "${workerName}")"
  control="$(controlPortOf "${workerName}")"
  handoff="$(handoffPortOf "${workerName}")"
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
    --engine-handoff-port "${handoff}"
  )

  if [[ "${role}" == "decode" ]]; then
    args+=(--table "${FIXTURE_DIR}/decode.pb" --control-port "${control}")
  else
    [[ -n "${peers}" ]] || die "prefill '${workerName}' has empty peers"
    args+=(--prefill-table "${FIXTURE_DIR}/prefill.pb")
    local tag
    IFS=',' read -ra peerTags <<<"${peers}"
    for tag in "${peerTags[@]}"; do
      [[ -n "${tag}" ]] && args+=(--peer "${tag}")
    done
  fi

  echo "[control-plane] ${workerName} peers='${peers}' health=${health}" \
       "control=${control} handoff=${handoff}"
  MC_TCP_BIND_ADDRESS="${MC_TCP_BIND_ADDRESS}" \
  KAFKA_BROKERS="${KAFKA_BROKERS}" \
  KAFKA_GROUP_ID="migration-workers-${workerName}" \
    "${WORKER_BIN}" "${args[@]}" >"${log}" 2>&1 &
  echo $! >"${STATE_DIR}/${workerName}.pid"

  pushDeviceMap "${workerName}"
}

waitReady() {
  local deadline=$(( SECONDS + READY_WAIT_SEC )) name port code
  echo "[control-plane] waiting up to ${READY_WAIT_SEC}s for /readyz..."
  while (( SECONDS < deadline )); do
    local allOk=1
    for name in $(allNames); do
      if ! kill -0 "$(cat "${STATE_DIR}/${name}.pid")" 2>/dev/null; then
        echo "[control-plane] worker ${name} died during bring-up" >&2
        tail -80 "${STATE_DIR}/${name}.log" >&2 || true
        return 1
      fi
      port="$(healthPortOf "${name}")"
      code="$(httpCode "${port}" /readyz)"
      if [[ "${code}" != "200" ]]; then
        allOk=0
        break
      fi
    done
    if (( allOk )); then
      echo "[control-plane] all workers /readyz=200"
      return 0
    fi
    sleep 1
  done
  echo "[control-plane] /readyz timeout after ${READY_WAIT_SEC}s" >&2
  local n
  for n in $(allNames); do
    echo "---- ${n} (tail) ----" >&2
    tail -60 "${STATE_DIR}/${n}.log" >&2 || true
  done
  return 1
}

assertLogs() {
  local name log prefillOk=0 exchangeOk=0
  local expectedExchanges=${NUM_DECODE}

  for name in $(allNames); do
    log="${STATE_DIR}/${name}.log"
    [[ -f "${log}" ]] || die "missing log ${log}"

    grep -qE "DeviceMap handoff received" "${log}" \
      || die "${name}: DeviceMap socket handoff did not complete"

    case "${name}" in
      decode-*)
        grep -qE "published control endpoint kv_control/${name}" "${log}" \
          || die "${name}: kv_control publish missing"
        grep -qE "READY: segment=" "${log}" \
          || die "${name}: READY missing"
        ;;
      prefill-*)
        prefillOk=1
        local got
        # Bring-up path: "TABLE_EXCHANGE with 'decode-N' ok (...)"
        got="$(grep -cE "TABLE_EXCHANGE with 'decode-[0-9]+' ok" "${log}" || true)"
        (( got >= expectedExchanges )) \
          || die "${name}: expected ${expectedExchanges} TABLE_EXCHANGE ok, got ${got}"
        grep -qE "READY: ${NUM_DECODE}/${NUM_DECODE} decode channels connected" "${log}" \
          || die "${name}: READY ${NUM_DECODE}/${NUM_DECODE} missing"
        exchangeOk=1
        ;;
    esac
  done

  (( prefillOk )) || die "no prefill worker asserted"
  (( exchangeOk )) || die "TABLE_EXCHANGE assert failed"

  echo "[control-plane] PASS: handoff + kv_control + TABLE_EXCHANGE + READY" \
       "(${NUM_PREFILL}P+${NUM_DECODE}D, no KV migrate)"
}

cleanup() {
  local exitCode=$?
  local pf
  if [[ -d "${STATE_DIR}" ]]; then
    for pf in "${STATE_DIR}"/*.pid; do
      [[ -f "${pf}" ]] || continue
      kill "$(cat "${pf}")" 2>/dev/null || true
    done
  fi
  # Sweep only this binary's metadata-tagged workers (never pkill the script).
  pkill -f "mooncake_kv_migration_worker --metadata http://${BIND}:${META_PORT}/metadata" \
    2>/dev/null || true
  # Keep logs on failure for CI artifacts / local debug.
  if (( exitCode == 0 )); then
    rm -rf "${STATE_DIR}"
  else
    echo "[control-plane] leaving logs in ${STATE_DIR} (exit=${exitCode})" >&2
  fi
}

main() {
  [[ -x "${WORKER_BIN}" ]] || die "${WORKER_BIN} not executable (build --mooncake --kafka --kv-table)"
  [[ -x "${HANDOFF_SENDER_BIN}" ]] || die "${HANDOFF_SENDER_BIN} not executable"
  command -v curl >/dev/null 2>&1 || die "curl required"
  command -v python3 >/dev/null 2>&1 || die "python3 required"
  ensureFixtures

  trap cleanup EXIT
  rm -rf "${STATE_DIR}"
  mkdir -p "${STATE_DIR}"

  echo "[control-plane] metadata on ${BIND}:${META_PORT}"
  HTTP_PORT="${META_PORT}" BIND_HOST="${BIND}" \
  PYTHON="${PYTHON:-$(command -v python3)}" \
    bash "${META_SERVER}" >"${STATE_DIR}/metadata.log" 2>&1 &
  echo $! >"${STATE_DIR}/metadata.pid"

  local ready=0 i
  for (( i = 0; i < 40; i++ )); do
    if probeMetadata "$(metaUri)"; then ready=1; break; fi
    sleep 0.25
  done
  (( ready )) || die "metadata service not ready (see ${STATE_DIR}/metadata.log)"

  # Decodes first so kv_control is published before prefills dial.
  local n
  for (( n = 0; n < NUM_DECODE; n++ )); do launchOne "decode-${n}"; done
  sleep 1
  for (( n = 0; n < NUM_PREFILL; n++ )); do launchOne "prefill-${n}"; done

  waitReady
  assertLogs
}

main "$@"
