#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# local_worker_lab.sh — bring up the migration-worker mesh on THIS host as one
# INDEPENDENT process per worker (each on its own health port), so you can kill a
# single worker and relaunch just that one and watch it recover. Unlike the
# mpirun harness (run_migration_workers_mpi.sh), killing one worker here does NOT
# abort the others — this mirrors how deploy_migration_workers.sh supervises a
# real cluster, but on a single box with per-worker ports.
#
# Workers use single-launch identity (rank 0 => prefill-0, ranks [NP, NP+ND) =>
# decode-*), talk only through the local discovery service, and each serves
# /healthz /readyz /metrics on HEALTH_PORT_BASE + global-rank.
#
# Usage:
#   ./local_worker_lab.sh up [--prefill N] [--decode N]   # metadata + all workers
#   ./local_worker_lab.sh status                          # PID + /healthz + /readyz
#   ./local_worker_lab.sh kill  NAME                       # SIGKILL one (e.g. decode-5)
#   ./local_worker_lab.sh restart NAME                     # relaunch just that one
#   ./local_worker_lab.sh down                             # stop everything
#
# Env overrides (defaults in parens): WORKER_BIN (./build/bringup_mooncake_worker),
# LAYER_START (0), LAYER_END (61), HOST_DRAM_BYTES (1 MiB), HEALTH_PORT_BASE
# (19100), META_PORT (18082), DISCOVERY_TIMEOUT_SEC (60).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly RANK_LAUNCH="${SCRIPT_DIR}/migration_worker_rank_launch.sh"
readonly META_SERVER="${SCRIPT_DIR}/../../integration/run_mooncake_metadata_server.sh"
readonly STATE_DIR="${STATE_DIR:-/tmp/tt_mc_lab}"
readonly BIND="127.0.0.1"

# Identity/config: env-overridable, persisted at `up` and reloaded by the other
# commands so name->rank->port math stays consistent across invocations.
NUM_PREFILL="${NUM_PREFILL:-1}"
NUM_DECODE="${NUM_DECODE:-16}"
WORKER_BIN="${WORKER_BIN:-./build/bringup_mooncake_worker}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-61}"
HOST_DRAM_BYTES="${HOST_DRAM_BYTES:-1048576}"
DISCOVERY_TIMEOUT_SEC="${DISCOVERY_TIMEOUT_SEC:-60}"
HEALTH_PORT_BASE="${HEALTH_PORT_BASE:-19100}"
META_PORT="${META_PORT:-18082}"

die() { echo "ERROR: $*" >&2; exit 2; }
metaUri() { echo "http://${BIND}:${META_PORT}/metadata"; }

# name -> global rank (single-launch ordering): prefill-p=p, decode-d=NP+d.
rankOf() {
  local n="$1"
  case "$n" in
    prefill-*) echo "${n#prefill-}" ;;
    decode-*) echo $(( NUM_PREFILL + ${n#decode-} )) ;;
    *) die "bad worker name '$n' (expected prefill-N or decode-N)" ;;
  esac
}
portOf() { echo $(( HEALTH_PORT_BASE + $(rankOf "$1") )); }

allNames() {
  local i
  for (( i = 0; i < NUM_PREFILL; i++ )); do echo "prefill-${i}"; done
  for (( i = 0; i < NUM_DECODE; i++ )); do echo "decode-${i}"; done
}

saveConfig() {
  cat >"${STATE_DIR}/config" <<EOF
NUM_PREFILL=${NUM_PREFILL}
NUM_DECODE=${NUM_DECODE}
WORKER_BIN=${WORKER_BIN}
LAYER_START=${LAYER_START}
LAYER_END=${LAYER_END}
HOST_DRAM_BYTES=${HOST_DRAM_BYTES}
DISCOVERY_TIMEOUT_SEC=${DISCOVERY_TIMEOUT_SEC}
HEALTH_PORT_BASE=${HEALTH_PORT_BASE}
META_PORT=${META_PORT}
EOF
}
loadConfig() { [[ -f "${STATE_DIR}/config" ]] && source "${STATE_DIR}/config"; }

# Launch one worker as its own backgrounded process; record its PID. rank_launch
# turns rank + NUM_* into the worker's name/peers/layer slice and exec's it, so a
# relaunch reproduces the same worker identity exactly.
# A SIGKILL'd worker never deregisters, so its mooncake/rpc_meta/<name> entry
# lingers and the discovery service rejects the restart with "Duplicate rpc_meta
# key not allowed". Clear it first so a relaunch registers cleanly. Harmless on a
# first launch (the key won't exist yet).
clearRpcMeta() {
  curl -sS -X DELETE "$(metaUri)?key=mooncake/rpc_meta/$1" >/dev/null 2>&1 || true
}

launchOne() {
  local name="$1" rank port log
  rank="$(rankOf "${name}")"
  port="$(portOf "${name}")"
  log="${STATE_DIR}/${name}.log"
  clearRpcMeta "${name}"
  OMPI_COMM_WORLD_RANK="${rank}" NUM_PREFILL="${NUM_PREFILL}" NUM_DECODE="${NUM_DECODE}" \
  WORKER_BIN="${WORKER_BIN}" METADATA="$(metaUri)" HOST_DRAM_BYTES="${HOST_DRAM_BYTES}" \
  DISCOVERY_TIMEOUT_SEC="${DISCOVERY_TIMEOUT_SEC}" LAYER_START="${LAYER_START}" \
  LAYER_END="${LAYER_END}" HEALTH_PORT="${port}" MC_TCP_BIND_ADDRESS="${BIND}" \
    bash "${RANK_LAUNCH}" >"${log}" 2>&1 &
  echo $! >"${STATE_DIR}/${name}.pid"
  echo "[lab] ${name} up (rank ${rank}, health ${BIND}:${port}, pid $!, log ${log})"
}

httpCode() { curl -so /dev/null -w '%{http_code}' --max-time 2 "http://${BIND}:$1$2" 2>/dev/null || echo "---"; }

cmdUp() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --prefill) NUM_PREFILL="$2"; shift 2 ;;
      --decode) NUM_DECODE="$2"; shift 2 ;;
      *) die "unknown up arg: $1" ;;
    esac
  done
  [[ -x "${WORKER_BIN}" ]] || die "${WORKER_BIN} not found/executable (build with ./build.sh --mooncake)"
  if [[ -f "${STATE_DIR}/metadata.pid" ]] && kill -0 "$(cat "${STATE_DIR}/metadata.pid")" 2>/dev/null; then
    die "lab already up (${STATE_DIR}); run '$0 down' first"
  fi
  mkdir -p "${STATE_DIR}"
  saveConfig
  echo "[lab] starting discovery service on ${BIND}:${META_PORT}"
  HTTP_PORT="${META_PORT}" BIND_HOST="${BIND}" "${META_SERVER}" >"${STATE_DIR}/metadata.log" 2>&1 &
  echo $! >"${STATE_DIR}/metadata.pid"
  sleep 2
  local n
  for n in $(allNames); do launchOne "${n}"; done
  echo "[lab] launched $(( NUM_PREFILL + NUM_DECODE )) worker(s). Try: $0 status"
}

cmdStatus() {
  loadConfig
  printf '%-12s %-10s %-14s %-9s %-8s\n' NAME PID HEALTH HEALTHZ READYZ
  local n pid port shown
  for n in $(allNames); do
    pid="$(cat "${STATE_DIR}/${n}.pid" 2>/dev/null || echo -)"
    port="$(portOf "${n}")"
    shown="dead"
    [[ "${pid}" != "-" ]] && kill -0 "${pid}" 2>/dev/null && shown="${pid}"
    printf '%-12s %-10s %-14s %-9s %-8s\n' \
      "${n}" "${shown}" "${BIND}:${port}" "$(httpCode "${port}" /healthz)" "$(httpCode "${port}" /readyz)"
  done
}

cmdKill() {
  loadConfig
  local name="${1:?usage: $0 kill NAME}" pf
  pf="${STATE_DIR}/${name}.pid"
  [[ -f "${pf}" ]] || die "no tracked pid for '${name}' (is the lab up?)"
  local pid; pid="$(cat "${pf}")"
  echo "[lab] SIGKILL ${name} (pid ${pid})"
  kill -9 "${pid}" 2>/dev/null || true
}

cmdRestart() {
  loadConfig
  local name="${1:?usage: $0 restart NAME}"
  echo "[lab] relaunching ${name}"
  launchOne "${name}"
}

cmdDown() {
  echo "[lab] tearing down"
  pkill -f "bringup_mooncake_worker --metadata" 2>/dev/null || true
  [[ -f "${STATE_DIR}/metadata.pid" ]] && kill "$(cat "${STATE_DIR}/metadata.pid")" 2>/dev/null || true
  rm -rf "${STATE_DIR}"
}

case "${1:-}" in
  up) shift; cmdUp "$@" ;;
  status) cmdStatus ;;
  kill) shift; cmdKill "$@" ;;
  restart) shift; cmdRestart "$@" ;;
  down) cmdDown ;;
  *) echo "usage: $0 {up [--prefill N] [--decode N]|status|kill NAME|restart NAME|down}"; exit 1 ;;
esac
