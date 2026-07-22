#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# deploy_migration_workers.sh — single-command deploy of Mooncake migration
# workers across an Exabox cluster.
#
# One invocation brings up the whole stack:
#   1. Starts the Mooncake HTTP discovery service on --discovery-host.
#   2. Launches one mooncake_kv_migration_worker per host, each as its OWN
#      process (local, or over ssh). One worker on each --prefill-host, one on
#      each --decode-host. Each worker's logical tag (prefill-<i> / decode-<i>,
#      or --{prefill,decode}-tags) is used as its --name, --host, and --peer key.
#   3. Workers find each other through the discovery service (register-then-
#      resolve) — a prefill resolves each decode tag to its routable host via the
#      metadata service and dials its control channel. No MPI/collectives.
#   4. Prefill workers consume Kafka (request/ack) and drive real KV migration
#      across the decode hosts; decode workers are passive control-servers.
#   5. Layer ownership lives INSIDE the decode table (.pb): each chunk's
#      fabric_node_host names the decode that owns it. --prefill-table /
#      --decode-table (or the config) supply the tables; --layer-start/--layer-end
#      are no longer used here.
#   6. A watchdog then polls every worker's /healthz (HTTP is the health signal).
#      After N consecutive misses it relaunches ONLY that worker in place. The
#      launch PID is only a kill handle for processes this script started — it
#      is NOT required for "healthy" (manual revive / dead ssh must not look
#      like a down worker). The watchdog loop also holds the deploy open so
#      Ctrl-C tears the stack down.
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
#   * the same mooncake_kv_migration_worker binary AND the KV .pb tables at the
#     same path on all hosts (NFS share, or rsync first), built with the health
#     server + Mooncake + kv-table; every worker host needs TT devices
#   * curl on this host (health/readiness probes) and python3 (discovery probe)
#
# Example — 2 prefill hosts + 4 decode hosts (tables + tags from the config):
#   ./scripts/deploy_migration_workers.sh \
#     --discovery-host bh-glx-c01u02 \
#     --prefill-hosts  bh-glx-c01u02,bh-glx-c01u03 \
#     --decode-hosts   bh-glx-c01u08,bh-glx-c02u02,bh-glx-c03u02,bh-glx-c04u02 \
#     --health-port 9109

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Per-worker launcher for the REAL data-plane worker (mooncake_kv_migration_worker).
readonly RANK_LAUNCH="${SCRIPT_DIR}/../tests/e2e/scripts/migration_worker_launch.sh"
readonly META_SERVER="${SCRIPT_DIR}/../tests/integration/run_mooncake_metadata_server.sh"
# Sourced (if present) before flag parsing, so --flags still override it. Holds
# the table-coupled settings (table paths, host tags, device map).
readonly DEFAULT_CONFIG="${SCRIPT_DIR}/migration_deploy.conf"

# --- defaults (override via flags) ---
DISCOVERY_HOST=""
PREFILL_HOSTS=""
DECODE_HOSTS=""
LAYER_START=0
LAYER_END=0
BUILD_DIR="./build"
WORKER_BIN=""
# The KV tables the real worker migrates (see migration_deploy.conf). Required.
PREFILL_TABLE="${PREFILL_TABLE:-}"
DECODE_TABLE="${DECODE_TABLE:-}"
# OPTIONAL directory of per-host FabricNode->UMD chip maps
# (<DEVICE_MAP_DIR>/<tag>.devmap). Unset => discovery-only e2e.
DEVICE_MAP_DIR="${DEVICE_MAP_DIR:-}"
# When DEVICE_MAP_DIR is set, deploy pushes each map over localhost
# (engine_handoff_sender → --engine-handoff-port) after the worker starts with
# its .pb file. 0 disables the socket path and passes --device-map as a file.
ENGINE_HANDOFF_PORT="${ENGINE_HANDOFF_PORT:-}"
HANDOFF_SENDER_BIN="${HANDOFF_SENDER_BIN:-}"
# Optional table host tags (fabric_node_host), aligned with the host CSVs.
# Empty => default to logical prefill-<i> / decode-<i>.
PREFILL_TAGS="${PREFILL_TAGS:-}"
DECODE_TAGS="${DECODE_TAGS:-}"
# Optional alternate config file (else DEFAULT_CONFIG next to this script).
CONFIG_FILE="${CONFIG_FILE:-}"
DISCOVERY_PORT=8080
# HTTP health port every worker exposes (/healthz /readyz /metrics). One worker
# per host, so they all share it. REQUIRED — the watchdog probes it.
HEALTH_PORT=0
# KV control-plane port each decode binds its control server on AND publishes to
# the metadata service (kv_control/<tag> -> host:CONTROL_PORT). One worker per
# host, so all decodes share it; prefill discovers each peer's host:port from
# metadata (this value is only the fallback for a peer that hasn't published).
# Set in migration_deploy.conf or override with --control-port.
CONTROL_PORT="${CONTROL_PORT:-18650}"
# OPTIONAL container name. When set, workers run inside this container (via a
# docker-exec --worker-bin wrapper) as root, so the sweep/teardown must kill them
# with `docker exec <CONTAINER> pkill` — a host-side pkill as the deploy user
# can't touch a root-in-container process, leaving stale workers + squatted ports
# (exabox Bug A). Empty => host-side pkill (bare-host/in-container deploys).
CONTAINER="${CONTAINER:-}"
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
# a silently-dead session within ~60s so a truly gone worker is detected. An
# array (not a string) so the flags word-split safely without relying on IFS.
SSH_OPTS=(-o ConnectTimeout=5 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
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
  --control-port PORT      KV control port a decode binds + publishes to
                           metadata (default ${CONTROL_PORT}); shared fleet-wide

Tables (required; from config or these flags). The real worker migrates KV
described by these .pb tables (layer ownership lives INSIDE the decode table's
fabric_node_host, not on the CLI):
  --config FILE            config to source first (default ${DEFAULT_CONFIG})
  --prefill-table PATH     prefill source table (.pb)
  --decode-table PATH      cluster decode table (.pb), shared by all decodes
  --device-map-dir DIR     OPTIONAL dir of per-host <tag>.devmap files ('mesh
                           chip umd' per line). With --engine-handoff-port
                           (default 18700 when this dir is set), deploy pushes
                           each map over localhost after the worker loads its
                           .pb; set --engine-handoff-port 0 for legacy
                           --device-map file mode. Omit dir for discovery-only
  --prefill-tags CSV       table host tags per prefill host (default prefill-<i>)
  --decode-tags CSV        table host tags per decode host (default decode-<i>)

Options:
  --build-dir PATH         cpp_server build dir (default ${BUILD_DIR})
  --worker-bin PATH        worker binary (default <build-dir>/mooncake_kv_migration_worker)
  --handoff-sender PATH    engine_handoff_sender (default <build-dir>/engine_handoff_sender)
  --engine-handoff-port N  DeviceMap socket port (default 18700 if
                           --device-map-dir set; 0 = file --device-map)
  --container NAME          workers run inside this container (docker-exec
                           --worker-bin wrapper); sweep/teardown kill via
                           'docker exec NAME pkill'. Omit for host-side pkill
  --discovery-port PORT    discovery service port (default ${DISCOVERY_PORT})
  --host-dram-bytes N      per-worker pool, page-aligned (default 4 GiB)
  --discovery-timeout-sec S  peer discovery timeout (default ${DISCOVERY_TIMEOUT_SEC})
  --kafka-brokers HOST:PORT  existing broker prefill workers use (default ${KAFKA_BROKERS})
  --poll-interval S        watchdog health poll period (default ${POLL_INTERVAL_SEC})
  --restart-after N        consecutive misses before a restart; 0 = monitor only
                           (default ${RESTART_AFTER})
  --sweep-grace S          grace before SIGTERM->SIGKILL on restart/teardown
                           (default ${SWEEP_GRACE_SEC})
  --layer-start N          accepted for compatibility; IGNORED — layer ownership
  --layer-end M            now lives in the decode table, not the CLI
  --dry-run                print the commands without launching anything
  -h, --help               this help
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }

# Source the config file BEFORE the full flag parse (so --flags override it).
# --config is honoured via an early scan; CONFIG_FILE env is a second override.
loadConfig() {
  local cfg="${DEFAULT_CONFIG}" prev="" a
  [[ -n "${CONFIG_FILE}" ]] && cfg="${CONFIG_FILE}"
  for a in "$@"; do
    [[ "${prev}" == "--config" ]] && cfg="${a}"
    prev="${a}"
  done
  if [[ -f "${cfg}" ]]; then
    echo "[deploy] loading config ${cfg}"
    # shellcheck disable=SC1090
    source "${cfg}"
  elif [[ "${cfg}" != "${DEFAULT_CONFIG}" ]]; then
    die "config file not found: ${cfg}"
  fi
}

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
      --config) CONFIG_FILE="$2"; shift 2 ;;
      --prefill-table) PREFILL_TABLE="$2"; shift 2 ;;
      --decode-table) DECODE_TABLE="$2"; shift 2 ;;
      --device-map-dir) DEVICE_MAP_DIR="$2"; shift 2 ;;
      --prefill-tags) PREFILL_TAGS="$2"; shift 2 ;;
      --decode-tags) DECODE_TAGS="$2"; shift 2 ;;
      --handoff-sender) HANDOFF_SENDER_BIN="$2"; shift 2 ;;
      --engine-handoff-port) ENGINE_HANDOFF_PORT="$2"; shift 2 ;;
      --discovery-port) DISCOVERY_PORT="$2"; shift 2 ;;
      --health-port) HEALTH_PORT="$2"; shift 2 ;;
      --control-port) CONTROL_PORT="$2"; shift 2 ;;
      --container) CONTAINER="$2"; shift 2 ;;
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
  [[ -z "${WORKER_BIN}" ]] && WORKER_BIN="${BUILD_DIR}/mooncake_kv_migration_worker"
  [[ -z "${HANDOFF_SENDER_BIN}" ]] && HANDOFF_SENDER_BIN="${BUILD_DIR}/engine_handoff_sender"
  [[ -n "${PREFILL_TABLE}" ]] || die "--prefill-table (or PREFILL_TABLE in config) is required"
  [[ -n "${DECODE_TABLE}" ]] || die "--decode-table (or DECODE_TABLE in config) is required"
  # Tables live on the NFS-shared path, so a local check on the lead is enough.
  [[ -f "${PREFILL_TABLE}" ]] || die "prefill table not found: ${PREFILL_TABLE}"
  [[ -f "${DECODE_TABLE}" ]] || die "decode table not found: ${DECODE_TABLE}"
  # Device maps are OPTIONAL: unset => discovery-only e2e (no transfer). When set
  # it's a real transfer run, so the dir must exist and every worker's own
  # <tag>.devmap is checked in addWorkerSlot (fail-fast before any launch).
  # Default: push maps over localhost (ENGINE_HANDOFF_PORT=18700). Set to 0 for
  # legacy --device-map file mode.
  if [[ -n "${DEVICE_MAP_DIR}" ]]; then
    [[ -d "${DEVICE_MAP_DIR}" ]] || die "device map dir not found: ${DEVICE_MAP_DIR}"
    if [[ -z "${ENGINE_HANDOFF_PORT}" ]]; then
      ENGINE_HANDOFF_PORT=18700
    fi
    if [[ "${ENGINE_HANDOFF_PORT}" != "0" ]]; then
      [[ -x "${HANDOFF_SENDER_BIN}" || -f "${HANDOFF_SENDER_BIN}" ]] || \
        die "engine_handoff_sender not found: ${HANDOFF_SENDER_BIN}"
    fi
  else
    ENGINE_HANDOFF_PORT=0
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
# index, host, table tag (== --name/--host/--peer), resolved <tag>.devmap,
# health port, launcher PID, and consecutive-failure count.
declare -a WK_ROLE=() WK_INDEX=() WK_HOST=() WK_TAG=() WK_DEVMAP=() WK_PORT=() WK_PID=() WK_FAILS=() WK_LOG=()
# Resolved peer CSV per worker (WK_PEERS[s]) — the generic, role-agnostic peer
# list this worker is launched with. See peersForWorker() for how it's derived.
declare -a WK_PEERS=()
# CSV of every decode's tag, in order — the default peer set for a prefill (fan-
# out is table-driven and may touch any decode; also used for TABLE_EXCHANGE).
DECODE_TAG_LIST=""
# Optional per-worker peer override, keyed by tag: WORKER_PEERS[<tag>]="p1,p2".
# A worker is just a migration worker with a peer list, so this lets you hand
# ANY worker (any role) an explicit peer set, overriding the role default. Set
# elements from the config file with plain assignment (no `declare`, which would
# be local to the sourcing function), e.g.  WORKER_PEERS[decode-0]="prefill-0".
declare -A WORKER_PEERS=()

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
    ssh "${SSH_OPTS[@]}" "${DISCOVERY_HOST}" \
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
# both the process is gone and the health port is free. When a worker name ($3)
# is given the match is scoped to exactly that worker (anchored so decode-1 never
# matches decode-15), so an unrelated deploy or a co-located worker is never hit;
# with no name it falls back to matching any migration worker on the host. Fed to
# `bash -s` on stdin (local or over ssh) so the pattern never lands in a
# process's argv — that both avoids self-matching the sweeper and keeps quoting
# sane.
sweepScript() {
  local port="$1" grace="$2" name="${3:-}"
  # Workers launched via docker-exec wrappers run as root with --pid=host.
  # A non-root `docker exec … pkill` gets Permission denied and leaves them
  # alive — always use -u root when CONTAINER is set.
  local run=""
  [[ -n "${CONTAINER}" ]] && run="docker exec -u root ${CONTAINER} "
  # Prefer exact binary name (avoids pkill -f matching the ssh/docker cmdline).
  # Optional --name scope for single-worker restart; teardown passes name too.
  cat <<EOF
killWorkers() {
  if [ -n '${name}' ]; then
    ${run}bash -lc 'pkill -9 -f "mooncake_kv_migration_worker.*--name ${name}([[:space:]]|\\\$)" 2>/dev/null || true'
  else
    ${run}bash -lc 'pids=\$(pgrep -x mooncake_kv_migration_worker 2>/dev/null); [ -n "\$pids" ] && kill -9 \$pids; true'
  fi
}
killWorkers
for _ in \$(seq 1 ${grace}); do
  if [ -n '${name}' ]; then
    ${run}bash -lc 'pgrep -af mooncake_kv_migration_worker 2>/dev/null | grep -q -- "--name ${name}"' || break
  else
    ${run}bash -lc 'pgrep -x mooncake_kv_migration_worker >/dev/null 2>&1' || break
  fi
  sleep 1
  killWorkers
done
if [ -n '${name}' ]; then
  if ${run}bash -lc 'pgrep -af mooncake_kv_migration_worker 2>/dev/null | grep -q -- "--name ${name}"'; then
    echo "SWEEP_FAIL: worker still alive on \$(hostname -s)" >&2; exit 1
  fi
elif ${run}bash -lc 'pgrep -x mooncake_kv_migration_worker >/dev/null 2>&1'; then
  echo "SWEEP_FAIL: worker still alive on \$(hostname -s)" >&2; exit 1
fi
if command -v ss >/dev/null 2>&1 && ss -ltnH 2>/dev/null | grep -q ":${port} "; then
  echo "SWEEP_FAIL: port ${port} still bound on \$(hostname -s)" >&2; exit 1
fi
echo "SWEEP_OK: ${port} free on \$(hostname -s)"
EOF
}

# Kill the migration worker on a host and BLOCK until its process is gone and the
# health port is free. Pass the worker name ($2) to scope the kill to exactly
# that worker; omit it to match any migration worker on the host. Returns
# non-zero if the port cannot be freed, so the caller refuses to relaunch into a
# squatted port (the churn bug).
sweepWorkerOnHost() {
  local host="$1" name="${2:-}" script
  script="$(sweepScript "${HEALTH_PORT}" "${SWEEP_GRACE_SEC}" "${name}")"
  if isLocalHost "${host}"; then
    bash -s <<<"${script}"
  else
    ssh "${SSH_OPTS[@]}" "${host}" bash -s <<<"${script}"
  fi
}

# A worker that died without deregistering leaves mooncake/rpc_meta/<name> (and
# often kv_control/<name>). Discovery rejects the restart with "Duplicate
# rpc_meta key not allowed" (TransferEngine::init fails). Clear both so a
# relaunch registers cleanly. Harmless on a first launch (keys won't exist).
# Manual revive MUST call this too — skipping it is a common bring-up failure.
clearRpcMeta() {
  # Short timeout — never hang teardown when metadata is already dead.
  curl -sS --max-time 1 -X DELETE "${META_URI}?key=mooncake/rpc_meta/$1" >/dev/null 2>&1 || true
  curl -sS --max-time 1 -X DELETE "${META_URI}?key=kv_control/$1" >/dev/null 2>&1 || true
  curl -sS --max-time 1 -X DELETE "${META_URI}?key=mooncake/ram/$1" >/dev/null 2>&1 || true
}

# Register one worker slot (role, role-local index, host, tag) in the arrays.
# When DEVICE_MAP_DIR is set, resolves this worker's host-specific map to
# <DEVICE_MAP_DIR>/<tag>.devmap and fails fast if it's missing (a wrong/absent
# map opens the wrong chip). Empty when unset => discovery-only, no transfer.
# Peer CSV for a worker, role-agnostic: an explicit WORKER_PEERS[<tag>] override
# always wins; otherwise the role default — a prefill peers with every decode
# (DECODE_TAG_LIST) for control TABLE_EXCHANGE + migrate; a decode peers with
# nothing (pure receiver). Prefill reads a complete DECODE_TAG_LIST because
# initWorkerSlots builds it before adding any prefill slot.
#
# IMPORTANT: decode control now multi-accepts (N prefills can TCP to one decode),
# but each prefill still uses a distinct Kafka group (broadcast). Every prefill
# would then attempt the same migration UUID against a shared decode — unsafe
# until Kafka ownership is exclusive. Default all-to-all is therefore only safe
# with NUM_PREFILL=1. With multiple prefills, set WORKER_PEERS so each decode
# appears in at most one prefill's peer list (assertExclusiveDecodePeers), or
# keep a single prefill.
peersForWorker() {
  local role="$1" tag="$2"
  if [[ -n "${WORKER_PEERS[$tag]:-}" ]]; then
    printf '%s' "${WORKER_PEERS[$tag]}"
  elif [[ "${role}" == "prefill" ]]; then
    printf '%s' "${DECODE_TAG_LIST}"
  fi
}

# Fail fast when two prefills would share the same decode peer. Control TCP
# multi-accept allows N sessions, but Kafka broadcast means every prefill
# consumes/acks the same migration UUID — shared decode peers stay unsafe until
# there is an explicit single Kafka owner. See migration_worker_rank_launch.sh
# for a round-robin WORKER_PEERS pattern.
assertExclusiveDecodePeers() {
  declare -A decodeOwner=()
  local s peers peer
  for (( s = 0; s < ${#WK_ROLE[@]}; s++ )); do
    [[ "${WK_ROLE[$s]}" == "prefill" ]] || continue
    IFS=',' read -ra peers <<<"${WK_PEERS[$s]}"
    for peer in "${peers[@]}"; do
      [[ -n "${peer}" ]] || continue
      if [[ -n "${decodeOwner[$peer]:-}" ]]; then
        die "decode peer '${peer}' assigned to both '${decodeOwner[$peer]}' and '${WK_TAG[$s]}'. \
Each prefill has its own Kafka group (broadcast), so two prefills cannot safely \
share a decode even though control TCP multi-accepts. Use one prefill, or set \
WORKER_PEERS so each decode has a single prefill owner \
(see migration_worker_rank_launch.sh round-robin)."
      fi
      decodeOwner[$peer]="${WK_TAG[$s]}"
    done
  done
}

addWorkerSlot() {
  local devmap=""
  if [[ -n "${DEVICE_MAP_DIR}" ]]; then
    devmap="${DEVICE_MAP_DIR}/$4.devmap"
    [[ -f "${devmap}" ]] || die "device map not found for tag '$4': ${devmap}"
  fi
  WK_ROLE+=("$1"); WK_INDEX+=("$2"); WK_HOST+=("$3"); WK_TAG+=("$4"); WK_DEVMAP+=("${devmap}")
  WK_PEERS+=("$(peersForWorker "$1" "$4")")
  WK_PORT+=("${HEALTH_PORT}"); WK_PID+=(""); WK_FAILS+=(0)
  WK_LOG+=("/tmp/tt_mc_deploy_$1-$2.log")
}

# Map worker i of each role onto host i of that role's CSV (one worker per host)
# and resolve its table tag: the i'th entry of --{prefill,decode}-tags, or the
# logical default {prefill,decode}-<i> when none is given.
initWorkerSlots() {
  local -a prefill_hosts decode_hosts prefill_tags decode_tags
  IFS=',' read -ra prefill_hosts <<<"${PREFILL_HOSTS}"
  IFS=',' read -ra decode_hosts <<<"${DECODE_HOSTS}"
  IFS=',' read -ra prefill_tags <<<"${PREFILL_TAGS}"
  IFS=',' read -ra decode_tags <<<"${DECODE_TAGS}"
  local i tag
  # Decodes first so DECODE_TAG_LIST is complete before any prefill reads it.
  for (( i = 0; i < NUM_DECODE; i++ )); do
    tag="${decode_tags[$i]:-decode-${i}}"
    DECODE_TAG_LIST="${DECODE_TAG_LIST:+${DECODE_TAG_LIST},}${tag}"
  done
  for (( i = 0; i < NUM_PREFILL; i++ )); do
    tag="${prefill_tags[$i]:-prefill-${i}}"
    addWorkerSlot "prefill" "${i}" "${prefill_hosts[$i]}" "${tag}"
  done
  for (( i = 0; i < NUM_DECODE; i++ )); do
    tag="${decode_tags[$i]:-decode-${i}}"
    addWorkerSlot "decode" "${i}" "${decode_hosts[$i]}" "${tag}"
  done
}

# The full env+command for the worker in slot $1. migration_worker_launch.sh
# turns this env into the real worker's flags (--name/--host/--table/--peer),
# so a relaunch reproduces the worker exactly. MC_TCP_BIND_ADDRESS=auto lets
# each host resolve its own routable IP for peers to reach it on. PEERS is this
# worker's resolved peer CSV (role-agnostic); the launcher forwards it as --peer.
workerCmd() {
  local s="$1" role="${WK_ROLE[$s]}" tag="${WK_TAG[$s]}" devmap="${WK_DEVMAP[$s]}"
  local peers="${WK_PEERS[$s]}"
  # Socket DeviceMap: pass ENGINE_HANDOFF_PORT (deploy pushes the .devmap after
  # start). File mode (port 0): pass DEVICE_MAP for --device-map. Discovery-only:
  # neither.
  local mapEnv=""
  if [[ "${ENGINE_HANDOFF_PORT}" != "0" && -n "${devmap}" ]]; then
    mapEnv="ENGINE_HANDOFF_PORT=${ENGINE_HANDOFF_PORT} "
  elif [[ -n "${devmap}" ]]; then
    mapEnv="DEVICE_MAP=${devmap} "
  fi
  printf '%s' "WORKER_ROLE=${role} WORKER_TAG=${tag} \
WORKER_BIN=${WORKER_BIN} METADATA=${META_URI} \
KAFKA_BROKERS=${KAFKA_BROKERS} HEALTH_PORT=${HEALTH_PORT} \
CONTROL_PORT=${CONTROL_PORT} \
${CONTAINER:+CTR=${CONTAINER} }\
PREFILL_TABLE=${PREFILL_TABLE} DECODE_TABLE=${DECODE_TABLE} \
${mapEnv}PEERS=${peers} \
MC_TCP_BIND_ADDRESS=auto bash ${RANK_LAUNCH}"
}

# Push <tag>.devmap to the worker on 127.0.0.1:ENGINE_HANDOFF_PORT (same box).
# Retries until the worker is listening after loading its .pb.
pushDeviceMapSlot() {
  local s="$1" host="${WK_HOST[$s]}" devmap="${WK_DEVMAP[$s]}"
  local tag="${WK_TAG[$s]}"
  [[ "${ENGINE_HANDOFF_PORT}" != "0" && -n "${devmap}" ]] || return 0
  local sendCmd="${HANDOFF_SENDER_BIN} --host 127.0.0.1 --port ${ENGINE_HANDOFF_PORT} --device-map ${devmap}"
  local attempt=0 maxAttempts=60
  while (( attempt < maxAttempts )); do
    if (( DRY_RUN )); then
      echo "[dry-run] push DeviceMap ${tag} on ${host}: ${sendCmd}"
      return 0
    fi
    if isLocalHost "${host}"; then
      if bash -c "${sendCmd}" >/dev/null 2>&1; then
        echo "[deploy] DeviceMap pushed to ${tag} on ${host} (port ${ENGINE_HANDOFF_PORT})"
        return 0
      fi
    else
      if ssh "${SSH_OPTS[@]}" "${host}" "${sendCmd}" >/dev/null 2>&1; then
        echo "[deploy] DeviceMap pushed to ${tag} on ${host} (port ${ENGINE_HANDOFF_PORT})"
        return 0
      fi
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
  die "failed to push DeviceMap to ${tag} on ${host} after ${maxAttempts}s (is worker listening on :${ENGINE_HANDOFF_PORT}?)"
}

# (Re)launch the worker in slot $1, locally or over ssh, tracking its PID as a
# kill handle only. For ssh the PID is the local ssh client (exits when the
# remote worker exits *if* that ssh session still owns it); for local hosts
# bash runs the worker in the background. Health probing uses /healthz, not
# this PID — see workerSlotHealthy.
launchWorkerSlot() {
  local s="$1" role="${WK_ROLE[$s]}" index="${WK_INDEX[$s]}" host="${WK_HOST[$s]}"
  local log="${WK_LOG[$s]}" cmd
  cmd="$(workerCmd "${s}")"
  clearRpcMeta "${WK_TAG[$s]}"
  : >"${log}"
  if isLocalHost "${host}"; then
    bash -c "${cmd}" >"${log}" 2>&1 &
  else
    ssh "${SSH_OPTS[@]}" "${host}" "${cmd}" >"${log}" 2>&1 &
  fi
  WK_PID[$s]=$!
  WK_FAILS[$s]=0
  echo "[deploy] ${role}-${index} on ${host} started (pid ${WK_PID[$s]}, log ${log})"
  pushDeviceMapSlot "${s}"
}

launchAllWorkers() {
  local s
  for (( s = 0; s < ${#WK_ROLE[@]}; s++ )); do launchWorkerSlot "${s}"; done
}

# Health = /healthz only. The launch PID is deploy's kill/sweep handle for a
# process *this script* started — not a liveness oracle. Requiring both made
# deploy lie after kill+manual-revive or a dead ssh client: /readyz true on the
# host, while supervise counted forever-unhealthy against a stale WK_PID
# (especially with --restart-after 0, which never refreshes the PID).
workerSlotHealthy() {
  local s="$1" pid="${WK_PID[$s]:-}" host="${WK_HOST[$s]}" port="${WK_PORT[$s]}"
  if ! curl -fsS --max-time 2 "http://${host}:${port}/healthz" >/dev/null 2>&1; then
    return 1
  fi
  if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
    echo "[deploy] ${WK_ROLE[$s]}-${WK_INDEX[$s]} on ${host}: /healthz ok but" \
         "launch pid ${pid} is gone (external revive or ssh drop) — adopting;" \
         "not unhealthy" >&2
    WK_PID[$s]=""
  fi
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
        # Clear metadata *before* kill/sweep so live prefills stop resolving this
        # peer while the host is torn down. launchWorkerSlot clears again before
        # relaunch (idempotent; covers SIGKILL'd workers that never unregistered).
        clearRpcMeta "${WK_TAG[$s]}"
        # Drop the local ssh handle, then hard-sweep the host. Only relaunch once

        # the port is confirmed free — otherwise a survivor squats it, the new
        # worker fails to bind, and the "restart" just churns launchers. If the
        # sweep can't free it, keep WK_FAILS so the next cycle retries the sweep.
        [[ -n "${WK_PID[$s]}" ]] && kill "${WK_PID[$s]}" 2>/dev/null
        if sweepWorkerOnHost "${WK_HOST[$s]}" "${WK_TAG[$s]}"; then
          launchWorkerSlot "${s}"
        else
          echo "[deploy] ERROR: could not free ${WK_HOST[$s]}:${HEALTH_PORT}; not relaunching (will retry next cycle)" >&2
        fi
      fi
    done
    sleep "${POLL_INTERVAL_SEC}"
  done
}

# One-shot teardown. Bash INT traps resume the interrupted loop unless we
# disable traps and exit — that was the "Ctrl-C forever / unhealthy spam" bug.
CLEANUP_DONE=0
cleanup() {
  local exitCode="${1:-0}"
  (( CLEANUP_DONE )) && return 0
  CLEANUP_DONE=1
  # Ignore further Ctrl-C so a signal mash cannot abort mid-sweep.
  trap "" INT TERM
  trap - EXIT

  echo ""
  echo "[deploy] tearing down..."
  local s
  # Best-effort meta clear (never blocks long — see clearRpcMeta).
  for (( s = 0; s < ${#WK_TAG[@]}; s++ )); do
    clearRpcMeta "${WK_TAG[$s]}"
  done
  for (( s = 0; s < ${#WK_PID[@]}; s++ )); do
    [[ -n "${WK_PID[$s]:-}" ]] && kill -9 "${WK_PID[$s]}" 2>/dev/null
  done
  [[ -n "${META_PID:-}" ]] && kill -9 "${META_PID}" 2>/dev/null

  # Hard-kill workers in parallel (root inside CONTAINER). Short grace — Ctrl-C
  # must finish in seconds, not hang on TERM waits.
  local savedGrace="${SWEEP_GRACE_SEC}"
  SWEEP_GRACE_SEC=2
  local -a sweepPids=() sweepLabels=()
  for (( s = 0; s < ${#WK_ROLE[@]}; s++ )); do
    sweepWorkerOnHost "${WK_HOST[$s]}" "${WK_TAG[$s]}" >/dev/null 2>&1 &
    sweepPids+=("$!")
    sweepLabels+=("${WK_ROLE[$s]}-${WK_INDEX[$s]}@${WK_HOST[$s]}")
  done
  local i waited
  for (( i = 0; i < ${#sweepPids[@]}; i++ )); do
    # Bound wait so a wedged ssh cannot trap Ctrl-C forever.
    # timeout(1) cannot wait on this shell's children — poll instead.
    waited=0
    while kill -0 "${sweepPids[$i]}" 2>/dev/null; do
      if (( waited >= 15 )); then
        kill -9 "${sweepPids[$i]}" 2>/dev/null || true
        echo "[deploy] WARN: sweep of ${sweepLabels[$i]} timed out" >&2
        break
      fi
      sleep 1
      waited=$((waited + 1))
    done
    wait "${sweepPids[$i]}" 2>/dev/null || \
      echo "[deploy] WARN: sweep of ${sweepLabels[$i]} failed" >&2
  done
  SWEEP_GRACE_SEC="${savedGrace}"
  echo "[deploy] tear down complete"
  exit "${exitCode}"
}

main() {
  loadConfig "$@"
  parseArgs "$@"
  validateArgs
  NUM_PREFILL="$(countHosts "${PREFILL_HOSTS}")"
  NUM_DECODE="$(countHosts "${DECODE_HOSTS}")"
  (( NUM_PREFILL >= 1 )) || die "--prefill-hosts must list at least one host"
  (( NUM_DECODE >= 1 )) || die "--decode-hosts must list at least one host"

  echo "[deploy] prefill hosts=${NUM_PREFILL} decode hosts=${NUM_DECODE} kafka=${KAFKA_BROKERS}"
  echo "[deploy] tables: prefill=${PREFILL_TABLE} decode=${DECODE_TABLE}${DEVICE_MAP_DIR:+ device-map-dir=${DEVICE_MAP_DIR}}"
  if [[ -z "${DEVICE_MAP_DIR}" ]]; then
    echo "[deploy] no device-map-dir: discovery-only (workers register + are discoverable, but cannot move KV)"
  elif [[ "${ENGINE_HANDOFF_PORT}" != "0" ]]; then
    echo "[deploy] DeviceMap via socket handoff port=${ENGINE_HANDOFF_PORT} sender=${HANDOFF_SENDER_BIN}"
  else
    echo "[deploy] DeviceMap via --device-map file (ENGINE_HANDOFF_PORT=0)"
  fi

  trap 'cleanup 130' INT TERM
  trap 'cleanup 0' EXIT
  startDiscoveryService || exit 1
  initWorkerSlots
  assertExclusiveDecodePeers

  if (( DRY_RUN )); then
    local s
    for (( s = 0; s < ${#WK_ROLE[@]}; s++ )); do
      echo "[dry-run] ${WK_HOST[$s]} (${WK_TAG[$s]}): $(workerCmd "${s}")"
      pushDeviceMapSlot "${s}"
    done
    echo "[deploy] dry-run complete"; CLEANUP_DONE=1; trap - EXIT INT TERM; exit 0
  fi

  launchAllWorkers
  waitReady || true
  superviseLoop
}

main "$@"
