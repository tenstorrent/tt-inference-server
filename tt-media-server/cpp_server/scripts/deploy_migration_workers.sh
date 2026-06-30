#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# deploy_migration_workers.sh — single-command deploy of Mooncake migration
# workers across an Exabox cluster.
#
# One invocation brings up the whole stack:
#   1. Starts the Mooncake HTTP discovery service on --discovery-host.
#   2. Launches one bringup_mooncake_worker per host via mpirun: one worker on
#      each --prefill-host, one on each --decode-host (first host in each list
#      is that role's rank-0 master).
#   3. Workers find each other through the discovery service (register-then-
#      resolve); rank-0 masters exchange their KV chunk address table.
#   4. Prefill workers connect to Kafka (request/ack); decode workers run with
#      --no-kafka.
#   5. The global KV layer range (--layer-start/--layer-end) is divided into one
#      contiguous slice per worker — NUM_PREFILL slices across prefill workers,
#      NUM_DECODE across decode — so each worker owns a distinct layer span, then
#      enters its busy loop, held up until this script is stopped.
#
# Kafka is NOT deployed here — the cluster's broker is assumed already running
# (point at it with --kafka-brokers, default kafka:9092).
#
# Requirements (mirroring the cross-host smokes):
#   * passwordless ssh from this host to --discovery-host and every worker host
#   * the same bringup_mooncake_worker binary at the same path on all hosts
#     (NFS share, or rsync the build dir first)
#   * Open MPI (mpirun) + the mooncake wheel on every host
#
# Example — 1 prefill host + 16 decode hosts, synthetic layer span 0..32:
#   ./scripts/deploy_migration_workers.sh \
#     --discovery-host bh-glx-c01u02 \
#     --prefill-hosts  bh-glx-c01u02 \
#     --decode-hosts   bh-glx-c01u08,bh-glx-c02u02,...,bh-glx-c09u02 \
#     --layer-start 0 --layer-end 32

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
# Mirrors bringup_mooncake_worker's K_DEFAULT_HOST_DRAM_BYTES (4 GiB). Kept in
# sync by hand; the worker also clamps/validates this against physical RAM.
HOST_DRAM_BYTES=$((4 * 1024 * 1024 * 1024))
DISCOVERY_TIMEOUT_SEC=60
KAFKA_BROKERS="kafka:9092"
PROTOCOL="tcp"
MIGRATION_NIC=""
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --discovery-host H --prefill-hosts CSV --decode-hosts CSV [options]

Required:
  --discovery-host HOST    host that runs the Mooncake discovery service
  --prefill-hosts CSV      prefill hosts (one worker each); first is rank-0 master
  --decode-hosts CSV       decode hosts (one worker each); first is rank-0 master

Options:
  --layer-start N          global model KV layer range start (default ${LAYER_START})
  --layer-end M            global range end (exclusive). The [start, end) span is
                           divided into NUM_PREFILL contiguous slices for prefill
                           and NUM_DECODE for decode, one per worker in rank order
                           (0=unset, default ${LAYER_END}, every worker owns all)
  --build-dir PATH         cpp_server build dir (default ${BUILD_DIR})
  --worker-bin PATH        worker binary (default <build-dir>/bringup_mooncake_worker)
  --discovery-port PORT    discovery service port (default ${DISCOVERY_PORT})
  --host-dram-bytes N      per-worker pool, page-aligned (default 4 GiB)
  --discovery-timeout-sec S  peer discovery timeout (default ${DISCOVERY_TIMEOUT_SEC})
  --kafka-brokers HOST:PORT  existing broker prefill workers use (default ${KAFKA_BROKERS})
  --protocol tcp|rdma      transport (default ${PROTOCOL})
  --migration-nic IFACE    pin Open MPI + Mooncake to this NIC/CIDR (multi-NIC hosts)
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
      --host-dram-bytes) HOST_DRAM_BYTES="$2"; shift 2 ;;
      --discovery-timeout-sec) DISCOVERY_TIMEOUT_SEC="$2"; shift 2 ;;
      --kafka-brokers) KAFKA_BROKERS="$2"; shift 2 ;;
      --protocol) PROTOCOL="$2"; shift 2 ;;
      --migration-nic) MIGRATION_NIC="$2"; shift 2 ;;
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
  [[ -z "${WORKER_BIN}" ]] && WORKER_BIN="${BUILD_DIR}/bringup_mooncake_worker"
  if (( LAYER_END != 0 )) && (( LAYER_END <= LAYER_START )); then
    die "--layer-end (${LAYER_END}) must be greater than --layer-start (${LAYER_START})"
  fi
  command -v mpirun >/dev/null 2>&1 || die "mpirun not found; install Open MPI"
}

# CSV -> count of non-empty fields.
countHosts() { awk -F',' '{n=0; for(i=1;i<=NF;i++) if($i!="") n++; print n}' <<<"$1"; }

# Is the given host name this machine? (covers localhost + this box's hostname)
isLocalHost() {
  local host="$1"
  [[ "${host}" == "localhost" || "${host}" == "127.0.0.1" || "${host}" == "$(hostname)" || "${host}" == "$(hostname -s)" ]]
}

# Write an Open MPI hostfile (one slot per host) from a CSV list.
writeHostfile() {
  local csv="$1" file="$2"
  : >"${file}"
  IFS=',' read -ra hosts <<<"${csv}"
  for host in "${hosts[@]}"; do
    [[ -n "${host}" ]] && echo "${host} slots=1" >>"${file}"
  done
}

META_URI=""
META_PID=""
PREFILL_PID=""
DECODE_PID=""
PREFILL_LOG="${PREFILL_LOG:-/tmp/tt_mc_deploy_prefill.log}"
DECODE_LOG="${DECODE_LOG:-/tmp/tt_mc_deploy_decode.log}"
META_LOG="${META_LOG:-/tmp/tt_mc_deploy_metadata.log}"
PREFILL_HOSTFILE="${PREFILL_HOSTFILE:-/tmp/tt_mc_deploy_prefill.hosts}"
DECODE_HOSTFILE="${DECODE_HOSTFILE:-/tmp/tt_mc_deploy_decode.hosts}"

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
    ssh "${DISCOVERY_HOST}" \
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

# Build the env-forwarding args (-x VAR) and leading exports shared by both
# mpirun launches. NIC pinning is applied only when --migration-nic is set.
mpirunForwardArgs() {
  local args=(-x WORKER_BIN -x METADATA -x HOST_DRAM_BYTES
              -x DISCOVERY_TIMEOUT_SEC -x NUM_PREFILL -x NUM_DECODE
              -x KAFKA_BROKERS -x LAYER_START -x LAYER_END
              -x MC_TCP_BIND_ADDRESS -x WORKER_ROLE)
  if [[ -n "${MIGRATION_NIC}" ]]; then
    args+=(--mca btl_tcp_if_include "${MIGRATION_NIC}"
           -x OMPI_MCA_btl_tcp_if_include="${MIGRATION_NIC}"
           -x PRTE_MCA_oob_tcp_if_include="${MIGRATION_NIC}"
           -x PMIX_MCA_ptl_tcp_if_include="${MIGRATION_NIC}")
  fi
  printf '%s\n' "${args[@]}"
}

launchRole() {
  local role="$1" hostfile="$2" np="$3" logfile="$4"
  mapfile -t fwd < <(mpirunForwardArgs)
  echo "[deploy] launching ${np} ${role} worker(s) via mpirun (hostfile ${hostfile})"
  if (( DRY_RUN )); then
    echo "[dry-run] WORKER_ROLE=${role} mpirun --hostfile ${hostfile} --map-by node --tag-output -np ${np} bash ${RANK_LAUNCH}"
    return 0
  fi
  : >"${logfile}"
  WORKER_ROLE="${role}" \
  WORKER_BIN="${WORKER_BIN}" METADATA="${META_URI}" \
  HOST_DRAM_BYTES="${HOST_DRAM_BYTES}" DISCOVERY_TIMEOUT_SEC="${DISCOVERY_TIMEOUT_SEC}" \
  NUM_PREFILL="${NUM_PREFILL}" NUM_DECODE="${NUM_DECODE}" \
  KAFKA_BROKERS="${KAFKA_BROKERS}" LAYER_START="${LAYER_START}" LAYER_END="${LAYER_END}" \
  MC_TCP_BIND_ADDRESS="auto" \
    mpirun "${fwd[@]}" --hostfile "${hostfile}" --map-by node --tag-output \
      -np "${np}" bash "${RANK_LAUNCH}" >"${logfile}" 2>&1 &
}

waitForMesh() {
  local total=$(( NUM_PREFILL + NUM_DECODE ))
  echo "[deploy] waiting up to ${DISCOVERY_TIMEOUT_SEC}s for ${total} worker(s) to connect..."
  local deadline=$(( SECONDS + DISCOVERY_TIMEOUT_SEC + 5 ))
  while (( SECONDS < deadline )); do
    local pc dc connected
    pc=$(grep -c "CONNECTED to" "${PREFILL_LOG}" 2>/dev/null) || pc=0
    dc=$(grep -c "CONNECTED to" "${DECODE_LOG}" 2>/dev/null) || dc=0
    connected=$(( pc + dc ))
    (( connected >= total )) && { echo "[deploy] all ${total} worker(s) CONNECTED"; return 0; }
    kill -0 "${PREFILL_PID}" 2>/dev/null || break
    kill -0 "${DECODE_PID}" 2>/dev/null || break
    sleep 1
  done
  echo "WARNING: not all workers reported CONNECTED; see ${PREFILL_LOG} / ${DECODE_LOG}" >&2
  return 1
}

cleanup() {
  echo ""
  echo "[deploy] tearing down..."
  [[ -n "${PREFILL_PID}" ]] && kill "${PREFILL_PID}" 2>/dev/null
  [[ -n "${DECODE_PID}" ]] && kill "${DECODE_PID}" 2>/dev/null
  [[ -n "${META_PID}" ]] && kill "${META_PID}" 2>/dev/null
  # Sweep straggler workers on every host (workers hold until SIGTERM).
  local all="${PREFILL_HOSTS},${DECODE_HOSTS}"
  IFS=',' read -ra hosts <<<"${all}"
  for host in "${hosts[@]}"; do
    [[ -z "${host}" ]] && continue
    if isLocalHost "${host}"; then
      pkill -f "bringup_mooncake_worker --metadata" 2>/dev/null || true
    else
      ssh "${host}" "pkill -f 'bringup_mooncake_worker --metadata'" 2>/dev/null || true
    fi
  done
}

main() {
  parseArgs "$@"
  validateArgs
  NUM_PREFILL="$(countHosts "${PREFILL_HOSTS}")"
  NUM_DECODE="$(countHosts "${DECODE_HOSTS}")"
  (( NUM_PREFILL >= 1 )) || die "--prefill-hosts must list at least one host"
  (( NUM_DECODE >= 1 )) || die "--decode-hosts must list at least one host"

  echo "[deploy] prefill hosts=${NUM_PREFILL} decode hosts=${NUM_DECODE} layers=[${LAYER_START},${LAYER_END}) kafka=${KAFKA_BROKERS}"
  writeHostfile "${PREFILL_HOSTS}" "${PREFILL_HOSTFILE}"
  writeHostfile "${DECODE_HOSTS}" "${DECODE_HOSTFILE}"

  trap cleanup EXIT INT TERM
  startDiscoveryService || exit 1

  launchRole "prefill" "${PREFILL_HOSTFILE}" "${NUM_PREFILL}" "${PREFILL_LOG}"
  PREFILL_PID=$!
  launchRole "decode" "${DECODE_HOSTFILE}" "${NUM_DECODE}" "${DECODE_LOG}"
  DECODE_PID=$!

  if (( DRY_RUN )); then echo "[deploy] dry-run complete"; trap - EXIT INT TERM; exit 0; fi

  waitForMesh || true
  echo "[deploy] workers are up and in their busy loop. Press Ctrl-C to tear down."
  # Hold the service up: block on the worker launches until interrupted.
  wait "${PREFILL_PID}" "${DECODE_PID}"
}

main "$@"
