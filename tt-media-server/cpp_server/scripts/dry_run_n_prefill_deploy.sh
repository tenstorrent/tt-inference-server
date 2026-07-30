#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Print (and optionally execute) an N-prefill × M-decode migration-worker
# deploy for exclusive Kafka ownership validation (#4795) without a model.
#
# Default: --dry-run (print docker/commands only). Pass --execute to actually
# call deploy_migration_workers.sh with --migration-mode dry-run.
#
# Examples:
#   ./dry_run_n_prefill_deploy.sh
#   ./dry_run_n_prefill_deploy.sh --prefill-hosts h1,h2 --decode-hosts d1,d2,d3,d4
#   ./dry_run_n_prefill_deploy.sh --execute --prefill-hosts "$(hostname)" ...
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy_migration_workers.sh"

NUM_PREFILL_HINT="${NUM_PREFILL:-2}"
NUM_DECODE_HINT="${NUM_DECODE:-4}"
PREFILL_HOSTS="${PREFILL_HOSTS:-}"
DECODE_HOSTS="${DECODE_HOSTS:-}"
DISCOVERY_SERVER="${DISCOVERY_SERVER:-$(hostname):8080}"
KAFKA_BROKERS="${KAFKA_BROKERS:-$(hostname):9092}"
PREFILL_TABLE="${PREFILL_TABLE:-/tmp/prefill_kv_chunk_table.pb}"
DECODE_TABLE="${DECODE_TABLE:-/tmp/decode-table.pb}"
EXECUTE=0

die() { echo "ERROR: $*" >&2; exit 2; }

usage() {
  cat <<EOF
usage: $0 [--execute] [--prefill-hosts CSV] [--decode-hosts CSV]
          [--discovery-server HOST:PORT] [--kafka-brokers HOST:PORT]
          [--prefill-table PATH] [--decode-table PATH]

Env defaults: NUM_PREFILL=${NUM_PREFILL_HINT} NUM_DECODE=${NUM_DECODE_HINT}
When --prefill-hosts / --decode-hosts are omitted, synthesizes placeholder
CSV lists sized from NUM_PREFILL / NUM_DECODE for --dry-run inspection only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --prefill-hosts) PREFILL_HOSTS="$2"; shift 2 ;;
    --decode-hosts) DECODE_HOSTS="$2"; shift 2 ;;
    --discovery-server) DISCOVERY_SERVER="$2"; shift 2 ;;
    --kafka-brokers) KAFKA_BROKERS="$2"; shift 2 ;;
    --prefill-table) PREFILL_TABLE="$2"; shift 2 ;;
    --decode-table) DECODE_TABLE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

if [[ -z "${PREFILL_HOSTS}" ]]; then
  PREFILL_HOSTS=""
  for (( i = 0; i < NUM_PREFILL_HINT; i++ )); do
    PREFILL_HOSTS="${PREFILL_HOSTS:+${PREFILL_HOSTS},}prefill-host-${i}"
  done
fi
if [[ -z "${DECODE_HOSTS}" ]]; then
  DECODE_HOSTS=""
  for (( i = 0; i < NUM_DECODE_HINT; i++ )); do
    DECODE_HOSTS="${DECODE_HOSTS:+${DECODE_HOSTS},}decode-host-${i}"
  done
fi

numPrefill=$(awk -F',' '{print NF}' <<<"${PREFILL_HOSTS}")
numDecode=$(awk -F',' '{print NF}' <<<"${DECODE_HOSTS}")

echo "[dry-run-n-prefill] topology: ${numPrefill} prefill × ${numDecode} decode"
echo "[dry-run-n-prefill] kafka exclusive: prefill i -> KAFKA_PARTITION=i"
echo "[dry-run-n-prefill] topics: ensure-partitions >= ${numPrefill}"
echo "[dry-run-n-prefill] peers: auto round-robin when NUM_PREFILL>1"
echo "[dry-run-n-prefill] discovery=${DISCOVERY_SERVER} kafka=${KAFKA_BROKERS}"

cmd=(
  bash "${DEPLOY}"
  --discovery-server "${DISCOVERY_SERVER}"
  --prefill-hosts "${PREFILL_HOSTS}"
  --decode-hosts "${DECODE_HOSTS}"
  --prefill-table "${PREFILL_TABLE}"
  --decode-table "${DECODE_TABLE}"
  --kafka-brokers "${KAFKA_BROKERS}"
  --migration-mode dry-run
)

if (( ! EXECUTE )); then
  cmd+=(--dry-run)
  echo "[dry-run-n-prefill] printing deploy commands (--execute to run for real)"
else
  echo "[dry-run-n-prefill] executing deploy with --migration-mode dry-run"
fi

printf '  %q' "${cmd[@]}"
echo
"${cmd[@]}"
