#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Per-rank launcher for the migration-worker MPI tests. mpirun starts one copy
# of this per rank; we turn the rank into a worker identity + its peer set and
# exec bringup_mooncake_worker. Two launch modes share the topology math:
#
#   1. Single-launch (#4294 MooncakeMpiDiscovery, default — WORKER_ROLE unset):
#      one mpirun spawns NUM_PREFILL + NUM_DECODE ranks. Rank assignment:
#        ranks [0, NUM_PREFILL)      -> prefill-0 .. prefill-(NUM_PREFILL-1)
#        ranks [NUM_PREFILL, total)  -> decode-0  .. decode-(NUM_DECODE-1)
#
#   2. Split-launch (MooncakeKafkaMigration — WORKER_ROLE=prefill|decode):
#      two mpirun invocations, one per role; each one's rank space is its own
#      role-local index. Prefill workers each get a UNIQUE KAFKA_GROUP_ID so a
#      single request fans out to all of them (one consumer per group =
#      broadcast); decode workers exec with --no-kafka.
#
#   Topology (same in both modes):
#     decode-d  peer  = prefill-(d % NUM_PREFILL)        (round-robin, one)
#     prefill-p peers = every decode-d with d % NUM_PREFILL == p
#
#   Round-robin keeps fan-out balanced and, as long as NUM_DECODE >=
#   NUM_PREFILL, guarantees every worker has >=1 peer — so `total` "CONNECTED"
#   lines proves the whole mesh wired up. (If NUM_PREFILL > NUM_DECODE the
#   surplus prefill workers get no peers and reach READY immediately.)
#
# Required env: WORKER_BIN, METADATA, HOST_DRAM_BYTES, DISCOVERY_TIMEOUT_SEC.
# Optional env: WORKER_ROLE (selects split-launch mode), NUM_PREFILL (default
# 4), NUM_DECODE (default 16), KAFKA_GROUP_ID_PREFIX (default
# `migration-workers-prefill`; one-group-per-rank suffix is appended).
set -euo pipefail

NUM_PREFILL="${NUM_PREFILL:-4}"
NUM_DECODE="${NUM_DECODE:-16}"
if (( NUM_PREFILL < 1 )); then
  echo "ERROR: NUM_PREFILL must be >= 1 (got ${NUM_PREFILL})" >&2
  exit 2
fi
readonly TOTAL_WORKERS=$(( NUM_PREFILL + NUM_DECODE ))
WORKER_ROLE="${WORKER_ROLE:-}"

rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}"
if [[ -z "${rank}" ]]; then
  echo "ERROR: no MPI rank in environment (OMPI_COMM_WORLD_RANK/PMI_RANK)" >&2
  exit 2
fi

: "${WORKER_BIN:?WORKER_BIN must point at bringup_mooncake_worker}"
: "${METADATA:?METADATA discovery URI required}"
: "${HOST_DRAM_BYTES:?HOST_DRAM_BYTES required}"
: "${DISCOVERY_TIMEOUT_SEC:?DISCOVERY_TIMEOUT_SEC required}"

# Resolve (role, role-local index) from rank under whichever launch mode is
# active, then defer the topology computation to the shared block below.
extra_args=()
case "${WORKER_ROLE}" in
  prefill)
    if (( rank >= NUM_PREFILL )); then
      echo "ERROR: WORKER_ROLE=prefill rank ${rank} out of range; expected -np ${NUM_PREFILL}." >&2
      exit 2
    fi
    role="prefill"
    role_index="${rank}"
    # One consumer group per prefill worker so a single request is broadcast
    # to every prefill (Kafka delivers each record to one member per group).
    KAFKA_GROUP_ID_PREFIX="${KAFKA_GROUP_ID_PREFIX:-migration-workers-prefill}"
    export KAFKA_GROUP_ID="${KAFKA_GROUP_ID_PREFIX}-${role_index}"
    ;;
  decode)
    if (( rank >= NUM_DECODE )); then
      echo "ERROR: WORKER_ROLE=decode rank ${rank} out of range; expected -np ${NUM_DECODE}." >&2
      exit 2
    fi
    role="decode"
    role_index="${rank}"
    extra_args+=(--no-kafka)
    ;;
  "")
    # Single-launch mode (existing MooncakeMpiDiscovery).
    if (( rank >= TOTAL_WORKERS )); then
      echo "ERROR: rank ${rank} out of range; NUM_PREFILL=${NUM_PREFILL} + NUM_DECODE=${NUM_DECODE} = ${TOTAL_WORKERS} workers. Launch mpirun with -np ${TOTAL_WORKERS}." >&2
      exit 2
    fi
    if (( rank < NUM_PREFILL )); then
      role="prefill"
      role_index="${rank}"
    else
      role="decode"
      role_index=$(( rank - NUM_PREFILL ))
    fi
    ;;
  *)
    echo "ERROR: unknown WORKER_ROLE='${WORKER_ROLE}' (expected: prefill, decode, or unset)" >&2
    exit 2
    ;;
esac

peer_args=()
if [[ "${role}" == "prefill" ]]; then
  name="prefill-${role_index}"
  for (( decode_index = 0; decode_index < NUM_DECODE; decode_index++ )); do
    if (( decode_index % NUM_PREFILL == role_index )); then
      peer_args+=(--peer "decode-${decode_index}")
    fi
  done
else
  name="decode-${role_index}"
  peer_args+=(--peer "prefill-$(( role_index % NUM_PREFILL ))")
fi

# ${peer_args[@]+...} keeps `set -u` happy when a prefill ends up with no peers.
exec "${WORKER_BIN}" \
  --metadata "${METADATA}" \
  --name "${name}" \
  ${peer_args[@]+"${peer_args[@]}"} \
  --host-dram-bytes "${HOST_DRAM_BYTES}" \
  --discovery-timeout-sec "${DISCOVERY_TIMEOUT_SEC}" \
  ${extra_args[@]+"${extra_args[@]}"}
