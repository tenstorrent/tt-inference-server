#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# deploy_migration_workers.sh — single-command deploy of Mooncake migration
# workers across an Exabox cluster.
#
# One invocation brings up the whole stack:
#   1. Starts the Mooncake HTTP discovery service on --discovery-host.
#   2. Launches one bringup_mooncake_worker per host, each as its OWN process
#      (local, or over ssh). One worker on each --prefill-host, one on each
#      --decode-host; index-0 (prefill-0 / decode-0) acts as its role's master.
#   3. Workers find each other through the discovery service (register-then-
#      resolve) — they do NO MPI/collective communication, so each is fully
#      independent. Masters exchange their KV chunk address table.
#   4. Prefill workers connect to Kafka (request/ack); decode workers run with
#      --no-kafka.
#   5. The global KV layer count (--layer-start/--layer-end) is padded up to the
#      next power of two (e.g. DeepSeek's 61 -> 64) and divided into one
#      contiguous slice per worker — NUM_PREFILL slices across prefill workers,
#      NUM_DECODE across decode — so each worker owns a distinct, aligned span.
#   6. A watchdog then polls every worker's /healthz and, when one dies or hangs,
#      relaunches ONLY that worker in place. It re-registers under the same name
#      and peers re-resolve it on their next request. The watchdog loop is also
#      what holds the deploy open, so the whole thing tears down on Ctrl-C.
#
# Why per-worker (not mpirun): the workers never talk over MPI — discovery is
# HTTP and KV moves over Mooncake TCP — so mpirun was only a launcher, and its
# "one dead rank aborts the whole job" behaviour actively prevents single-worker
# recovery. Launching each worker independently is simpler and supervisable.
#
# Kafka is NOT deployed here — the cluster's broker is assumed already running
# (point at it with --kafka-brokers, default kafka:9092).
#
# Requirements:
#   * passwordless ssh from this host to --discovery-host and every worker host
#   * the same bringup_mooncake_worker binary at the same path on all hosts
#     (NFS share, or rsync the build dir first), built with the health server
#   * curl on this host (health/readiness probes) and python3 (discovery probe)
#
# Example — 2 prefill hosts + 4 decode hosts, DeepSeek's 61 layers:
#   ./scripts/deploy_migration_workers.sh \
#     --discovery-host bh-glx-c01u02 \
#     --prefill-hosts  bh-glx-c01u02,bh-glx-c01u03 \
#     --decode-hosts   bh-glx-c01u08,bh-glx-c02u02,bh-glx-c03u02,bh-glx-c04u02 \
#     --layer-start 0 --layer-end 61 --health-port 9109

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly RANK_LAUNCH="${SCRIPT_DIR}/../tests/e2e/scripts/migration_worker_rank_launch.sh"
readonly META_SERVER="${SCRIPT_DIR}/../tests/integration/run_mooncake_metadata_server.sh"

# --- defaults (override via flags) ---
DISCOVERY_HOST=""
PREFILL_HOSTS=""
DECODE_HOSTS=""
LAYER_START=0
LAYER_END=0
BUILD_DIR="./build"
WORKER_BIN=""
DISCOVERY_PORT=8080
# HTTP health port every worker exposes (/healthz /readyz /metrics). One worker
# per host, so they all share it. REQUIRED — the watchdog probes it.
HEALTH_PORT=0
# Mirrors bringup_mooncake_worker's K_DEFAULT_HOST_DRAM_BYTES (4 GiB). Kept in
# sync by hand; the worker also clamps/validates this against physical RAM.
HOST_DRAM_BYTES=$((4 * 1024 * 1024 * 1024))
DISCOVERY_TIMEOUT_SEC=60
KAFKA_BROKERS="kafka:9092"
DRY_RUN=0
# Watchdog: poll each worker's /healthz every POLL_INTERVAL_SEC; after
# RESTART_AFTER consecutive misses, relaunch that worker. RESTART_AFTER=0 means
# monitor + log only (never relaunch), for debugging a crash in place.
POLL_INTERVAL_SEC=5
RESTART_AFTER=3
# ssh hardening: fail fast on an unreachable host (never hang a prompt) and drop
# a silently-dead session within ~60s so a truly gone worker is detected.
SSH_OPTS="-o ConnectTimeout=5 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
# A restart escalates SIGTERM -> (grace) -> SIGKILL and only relaunches once the
# health port is confirmed free, so a squatting worker can never wedge recovery.
SWEEP_GRACE_SEC=5

usage() {
  cat <<EOF
Usage: $(basename "$0") --discovery-host H --prefill-hosts CSV --decode-hosts CSV --health-port PORT [options]

Required:
  --discovery-host HOST    host that runs the Mooncake discovery service
  --prefill-hosts CSV      prefill hosts (one worker each); first is the master
  --decode-hosts CSV       decode hosts (one worker each); first is the master
  --health-port PORT       HTTP health port each worker serves (/healthz /readyz
                           /metrics); the watchdog probes it

Options:
  --layer-start N          global model KV layer range start (default ${LAYER_START})
  --layer-end M            global range end (exclusive), the model's REAL layer
                           count. The count is padded up to the next power of two
                           (e.g. DeepSeek's 61 -> 64) then divided into
                           NUM_PREFILL contiguous slices for prefill and
                           NUM_DECODE for decode, one per worker in index order
                           (0=unset, default ${LAYER_END}, every worker owns all)
  --build-dir PATH         cpp_server build dir (default ${BUILD_DIR})
  --worker-bin PATH        worker binary (default <build-dir>/bringup_mooncake_worker)
  --discovery-port PORT    discovery service port (default ${DISCOVERY_PORT})
  --host-dram-bytes N      per-worker pool, page-aligned (default 4 GiB)
  --discovery-timeout-sec S  peer discovery timeout (default ${DISCOVERY_TIMEOUT_SEC})
  --kafka-brokers HOST:PORT  existing broker prefill workers use (default ${KAFKA_BROKERS})
  --poll-interval S        watchdog health poll period (default ${POLL_INTERVAL_SEC})
  --restart-after N        consecutive misses before a restart; 0 = monitor only
                           (default ${RESTART_AFTER})
  --sweep-grace S          grace before SIGTERM->SIGKILL on restart/teardown
                           (default ${SWEEP_GRACE_SEC})
  --dry-run                print the commands without launching anything
  -h, --help               this help
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }

parseArgs() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --discovery-host) DISCOVERY_HOST="$2"; shift 2 ;;
      --prefill-hosts) PREFILL_HOSTS="$2"; shift 2 ;;
      --decode-hosts) DECODE_HOSTS="$2"; shift 2 ;;
      --layer-start) LAYER_START="$2"; shift 2 ;;
      --layer-end) LAYER_END="$2"; shift 2 ;;
      --build-dir) BUILD_DIR="$2"; shift 2 ;;
      --worker-bin) WORKER_BIN="$2"; shift 2 ;;
      --discovery-port) DISCOVERY_PORT="$2"; shift 2 ;;
      --health-port) HEALTH_PORT="$2"; shift 2 ;;
      --host-dram-bytes) HOST_DRAM_BYTES="$2"; shift 2 ;;
      --discovery-timeout-sec) DISCOVERY_TIMEOUT_SEC="$2"; shift 2 ;;
      --kafka-brokers) KAFKA_BROKERS="$2"; shift 2 ;;
      --poll-interval) POLL_INTERVAL_SEC="$2"; shift 2 ;;
      --restart-after) RESTART_AFTER="$2"; shift 2 ;;
      --sweep-grace) SWEEP_GRACE_SEC="$2"; shift 2 ;;
      --dry-run) DRY_RUN=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
}

validateArgs() {
  [[ -n "${DISCOVERY_HOST}" ]] || die "--discovery-host is required"
  [[ -n "${PREFILL_HOSTS}" ]] || die "--prefill-hosts is required"
  [[ -n "${DECODE_HOSTS}" ]] || die "--decode-hosts is required"
  [[ "${HEALTH_PORT}" != "0" ]] || die "--health-port is required (the watchdog probes it)"
  [[ -z "${WORKER_BIN}" ]] && WORKER_BIN="${BUILD_DIR}/bringup_mooncake_worker"
  if (( LAYER_END != 0 )) && (( LAYER_END <= LAYER_START )); then
    die "--layer-end (${LAYER_END}) must be greater than --layer-start (${LAYER_START})"
  fi
  command -v curl >/dev/null 2>&1 || die "curl not found; needed for health/readiness probes"
  command -v python3 >/dev/null 2>&1 || die "python3 not found; needed to probe the discovery service"
}

# CSV -> count of non-empty fields.
countHosts() { awk -F',' '{n=0; for(i=1;i<=NF;i++) if($i!="") n++; print n}' <<<"$1"; }

# Is the given host name this machine? (covers localhost + this box's hostname)
isLocalHost() {
  local host="$1"
  [[ "${host}" == "localhost" || "${host}" == "127.0.0.1" || "${host}" == "$(hostname)" || "${host}" == "$(hostname -s)" ]]
}

META_URI=""
META_PID=""
META_LOG="${META_LOG:-/tmp/tt_mc_deploy_metadata.log}"
# Per-worker tracking. Parallel arrays, one entry per worker: role, role-local
# index, host, health port, launcher PID, and consecutive-failure count.
declare -a WK_ROLE=() WK_INDEX=() WK_HOST=() WK_PORT=() WK_PID=() WK_FAILS=() WK_LOG=()

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

startDiscoveryService() {
  META_URI="http://${DISCOVERY_HOST}:${DISCOVERY_PORT}/metadata"
  echo "[deploy] starting discovery service on ${DISCOVERY_HOST}:${DISCOVERY_PORT}"
  if (( DRY_RUN )); then
    echo "[dry-run] HTTP_PORT=${DISCOVERY_PORT} BIND_HOST=0.0.0.0 ${META_SERVER} (on ${DISCOVERY_HOST})"
    return 0
  fi

  if isLocalHost "${DISCOVERY_HOST}"; then
    HTTP_PORT="${DISCOVERY_PORT}" BIND_HOST="0.0.0.0" \
      "${META_SERVER}" >"${META_LOG}" 2>&1 &
  else
    ssh ${SSH_OPTS} "${DISCOVERY_HOST}" \
      "HTTP_PORT='${DISCOVERY_PORT}' BIND_HOST='0.0.0.0' bash '${META_SERVER}'" \
      >"${META_LOG}" 2>&1 &
  fi
  META_PID=$!

  for _ in $(seq 1 20); do
    probeMetadata "${META_URI}" && { echo "[deploy] discovery service ready at ${META_URI}"; return 0; }
    sleep 0.5
  done
  echo "ERROR: discovery service not ready at ${META_URI}" >&2
  cat "${META_LOG}" >&2 || true
  return 1
}

# Emit the shell program a sweep runs on a host: SIGTERM the worker, wait up to
# `grace` seconds for a clean exit, SIGKILL any survivor, then FAIL loudly unless
# both the process is gone and the health port is free. Fed to `bash -s` on stdin
# (local or over ssh) so the pattern never lands in a process's argv — that both
# avoids self-matching the sweeper and keeps quoting sane.
sweepScript() {
  local port="$1" grace="$2"
  cat <<EOF
pat='bringup_mooncake_worker --metadata'
pkill -TERM -f "\$pat" 2>/dev/null || true
for _ in \$(seq 1 ${grace}); do pgrep -f "\$pat" >/dev/null 2>&1 || break; sleep 1; done
pkill -KILL -f "\$pat" 2>/dev/null || true
sleep 1
if pgrep -f "\$pat" >/dev/null 2>&1; then
  echo "SWEEP_FAIL: worker still alive on \$(hostname -s)" >&2; exit 1
fi
if command -v ss >/dev/null 2>&1 && ss -ltnH 2>/dev/null | grep -q ":${port} "; then
  echo "SWEEP_FAIL: port ${port} still bound on \$(hostname -s)" >&2; exit 1
fi
echo "SWEEP_OK: ${port} free on \$(hostname -s)"
EOF
}

# Kill the migration worker on a host and BLOCK until its process is gone and the
# health port is free. Returns non-zero if the port cannot be freed, so the caller
# refuses to relaunch into a squatted port (the churn bug). One worker per host,
# so matching on the binary is unambiguous.
sweepWorkerOnHost() {
  local host="$1" script
  script="$(sweepScript "${HEALTH_PORT}" "${SWEEP_GRACE_SEC}")"
  if isLocalHost "${host}"; then
    bash -s <<<"${script}"
  else
    ssh ${SSH_OPTS} "${host}" bash -s <<<"${script}"
  fi
}

# A worker that died without deregistering leaves a mooncake/rpc_meta/<name>
# entry, and the discovery service rejects the restart with "Duplicate rpc_meta
# key not allowed" (TransferEngine::init fails). Clear it so the relaunch
# registers cleanly. Harmless on a first launch (the key won't exist yet).
clearRpcMeta() {
  curl -sS -X DELETE "${META_URI}?key=mooncake/rpc_meta/$1" >/dev/null 2>&1 || true
}

# Register one worker slot (role, role-local index, host) in the parallel arrays.
addWorkerSlot() {
  WK_ROLE+=("$1"); WK_INDEX+=("$2"); WK_HOST+=("$3")
  WK_PORT+=("${HEALTH_PORT}"); WK_PID+=(""); WK_FAILS+=(0)
  WK_LOG+=("/tmp/tt_mc_deploy_$1-$2.log")
}

# Map worker i of each role onto host i of that role's CSV (one worker per host).
initWorkerSlots() {
  local -a prefill_hosts decode_hosts
  IFS=',' read -ra prefill_hosts <<<"${PREFILL_HOSTS}"
  IFS=',' read -ra decode_hosts <<<"${DECODE_HOSTS}"
  local i
  for (( i = 0; i < NUM_PREFILL; i++ )); do addWorkerSlot "prefill" "${i}" "${prefill_hosts[$i]}"; done
  for (( i = 0; i < NUM_DECODE; i++ )); do addWorkerSlot "decode" "${i}" "${decode_hosts[$i]}"; done
}

# The full env+command for one worker. rank_launch turns (WORKER_ROLE, rank,
# NUM_*) into the worker's name/peers/layer slice/health port and exec's the
# binary, so a relaunch reproduces the worker exactly. MC_TCP_BIND_ADDRESS=auto
# lets each host resolve its own routable IP for peers to reach it on.
workerCmd() {
  local role="$1" index="$2"
  printf '%s' "WORKER_ROLE=${role} OMPI_COMM_WORLD_RANK=${index} \
NUM_PREFILL=${NUM_PREFILL} NUM_DECODE=${NUM_DECODE} \
WORKER_BIN=${WORKER_BIN} METADATA=${META_URI} \
HOST_DRAM_BYTES=${HOST_DRAM_BYTES} DISCOVERY_TIMEOUT_SEC=${DISCOVERY_TIMEOUT_SEC} \
KAFKA_BROKERS=${KAFKA_BROKERS} LAYER_START=${LAYER_START} LAYER_END=${LAYER_END} \
HEALTH_PORT=${HEALTH_PORT} MC_TCP_BIND_ADDRESS=auto bash ${RANK_LAUNCH}"
}

# (Re)launch the worker in slot $1, locally or over ssh, tracking its PID. For
# ssh the PID is the local ssh client: it exits when the remote worker does, so
# it is a valid liveness handle; for local hosts bash exec's into the worker so
# the PID is the worker itself.
launchWorkerSlot() {
  local s="$1" role="${WK_ROLE[$s]}" index="${WK_INDEX[$s]}" host="${WK_HOST[$s]}"
  local log="${WK_LOG[$s]}" cmd
  cmd="$(workerCmd "${role}" "${index}")"
  clearRpcMeta "${role}-${index}"
  : >"${log}"
  if isLocalHost "${host}"; then
    bash -c "${cmd}" >"${log}" 2>&1 &
  else
    ssh ${SSH_OPTS} "${host}" "${cmd}" >"${log}" 2>&1 &
  fi
  WK_PID[$s]=$!
  WK_FAILS[$s]=0
  echo "[deploy] ${role}-${index} on ${host} started (pid ${WK_PID[$s]}, log ${log})"
}

launchAllWorkers() {
  local s
  for (( s = 0; s < ${#WK_ROLE[@]}; s++ )); do launchWorkerSlot "${s}"; done
}

# A slot is healthy if its launcher PID is alive AND /healthz answers 2xx — the
# HTTP check is what catches a hung-but-alive worker the PID check would miss.
workerSlotHealthy() {
  local s="$1" pid="${WK_PID[$s]}" host="${WK_HOST[$s]}" port="${WK_PORT[$s]}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null || return 1
  curl -fsS --max-time 2 "http://${host}:${port}/healthz" >/dev/null 2>&1 || return 1
  return 0
}

# Block until every worker's /readyz is 2xx or the discovery window elapses.
waitReady() {
  local total="${#WK_ROLE[@]}" deadline=$(( SECONDS + DISCOVERY_TIMEOUT_SEC + 10 ))
  while (( SECONDS < deadline )); do
    local ready=0 s
    for (( s = 0; s < total; s++ )); do
      curl -fsS --max-time 2 "http://${WK_HOST[$s]}:${WK_PORT[$s]}/readyz" \
        >/dev/null 2>&1 && ready=$(( ready + 1 ))
    done
    (( ready >= total )) && { echo "[deploy] all ${total} worker(s) READY"; return 0; }
    sleep 2
  done
  echo "WARNING: not all workers READY within ${DISCOVERY_TIMEOUT_SEC}s" >&2
  return 1
}

# Poll every worker; after RESTART_AFTER consecutive misses, sweep the host and
# relaunch just that worker. Also the deploy's hold-open loop: runs until
# interrupted. RESTART_AFTER=0 keeps monitoring/logging but never relaunches.
superviseLoop() {
  local action="restart after ${RESTART_AFTER} misses"
  (( RESTART_AFTER == 0 )) && action="monitor only (no restart)"
  echo "[deploy] supervising ${#WK_ROLE[@]} worker(s) every ${POLL_INTERVAL_SEC}s; ${action}. Ctrl-C to tear down."
  while true; do
    local s
    for (( s = 0; s < ${#WK_ROLE[@]}; s++ )); do
      if workerSlotHealthy "${s}"; then WK_FAILS[$s]=0; continue; fi
      WK_FAILS[$s]=$(( WK_FAILS[$s] + 1 ))
      echo "[deploy] ${WK_ROLE[$s]}-${WK_INDEX[$s]} on ${WK_HOST[$s]} unhealthy" \
           "(${WK_FAILS[$s]}/${RESTART_AFTER:-inf})" >&2
      if (( RESTART_AFTER > 0 && WK_FAILS[$s] >= RESTART_AFTER )); then
        echo "[deploy] restarting ${WK_ROLE[$s]}-${WK_INDEX[$s]} on ${WK_HOST[$s]}" >&2
        # Drop the local ssh handle, then hard-sweep the host. Only relaunch once
        # the port is confirmed free — otherwise a survivor squats it, the new
        # worker fails to bind, and the "restart" just churns launchers. If the
        # sweep can't free it, keep WK_FAILS so the next cycle retries the sweep.
        [[ -n "${WK_PID[$s]}" ]] && kill "${WK_PID[$s]}" 2>/dev/null
        if sweepWorkerOnHost "${WK_HOST[$s]}"; then
          launchWorkerSlot "${s}"
        else
          echo "[deploy] ERROR: could not free ${WK_HOST[$s]}:${HEALTH_PORT}; not relaunching (will retry next cycle)" >&2
        fi
      fi
    done
    sleep "${POLL_INTERVAL_SEC}"
  done
}

cleanup() {
  echo ""
  echo "[deploy] tearing down..."
  local s
  for (( s = 0; s < ${#WK_PID[@]}; s++ )); do
    [[ -n "${WK_PID[$s]:-}" ]] && kill "${WK_PID[$s]}" 2>/dev/null
  done
  [[ -n "${META_PID}" ]] && kill "${META_PID}" 2>/dev/null
  # Sweep straggler workers on every host in parallel (each blocks up to the
  # grace period; serial across 17 hosts would make teardown crawl).
  local all="${PREFILL_HOSTS},${DECODE_HOSTS}"
  IFS=',' read -ra hosts <<<"${all}"
  for host in "${hosts[@]}"; do
    [[ -z "${host}" ]] && continue
    sweepWorkerOnHost "${host}" >/dev/null 2>&1 &
  done
  wait
}

main() {
  parseArgs "$@"
  validateArgs
  NUM_PREFILL="$(countHosts "${PREFILL_HOSTS}")"
  NUM_DECODE="$(countHosts "${DECODE_HOSTS}")"
  (( NUM_PREFILL >= 1 )) || die "--prefill-hosts must list at least one host"
  (( NUM_DECODE >= 1 )) || die "--decode-hosts must list at least one host"

  echo "[deploy] prefill hosts=${NUM_PREFILL} decode hosts=${NUM_DECODE} layers=[${LAYER_START},${LAYER_END}) kafka=${KAFKA_BROKERS}"

  trap cleanup EXIT INT TERM
  startDiscoveryService || exit 1
  initWorkerSlots

  if (( DRY_RUN )); then
    local s
    for (( s = 0; s < ${#WK_ROLE[@]}; s++ )); do
      echo "[dry-run] ${WK_HOST[$s]}: $(workerCmd "${WK_ROLE[$s]}" "${WK_INDEX[$s]}")"
    done
    echo "[deploy] dry-run complete"; trap - EXIT INT TERM; exit 0
  fi

  launchAllWorkers
  waitReady || true
  superviseLoop
}

main "$@"
