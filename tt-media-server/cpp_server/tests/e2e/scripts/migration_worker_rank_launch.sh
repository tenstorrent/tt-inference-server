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
# `migration-workers-prefill`; one-group-per-rank suffix is appended),
# MC_TCP_BIND_ADDRESS (unset/"auto" => detect this host's primary IP),
# LAYER_START / LAYER_END (global model layer span; divided into NUM_PREFILL
# contiguous slices for prefill workers and NUM_DECODE for decode workers, each
# worker getting its role_index'th slice in order; unset/0 => worker owns all).
set -euo pipefail

# Single exit path for every unrecoverable misconfiguration: log to stderr with
# a uniform ERROR: prefix and exit 2. Keeps error handling consistent so a
# worker never limps on in a broken state and surfaces an obscure failure later.
die() {
  echo "ERROR: $*" >&2
  exit 2
}

NUM_PREFILL="${NUM_PREFILL:-4}"
NUM_DECODE="${NUM_DECODE:-16}"
if (( NUM_PREFILL < 1 )); then
  die "NUM_PREFILL must be >= 1 (got ${NUM_PREFILL})"
fi
readonly TOTAL_WORKERS=$(( NUM_PREFILL + NUM_DECODE ))
WORKER_ROLE="${WORKER_ROLE:-}"

rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}"
if [[ -z "${rank}" ]]; then
  die "no MPI rank in environment (OMPI_COMM_WORLD_RANK/PMI_RANK)"
fi

[[ -n "${WORKER_BIN:-}" ]] || die "WORKER_BIN must point at bringup_mooncake_worker"
[[ -n "${METADATA:-}" ]] || die "METADATA discovery URI required"
[[ -n "${HOST_DRAM_BYTES:-}" ]] || die "HOST_DRAM_BYTES required"
[[ -n "${DISCOVERY_TIMEOUT_SEC:-}" ]] || die "DISCOVERY_TIMEOUT_SEC required"

# Cross-host deployment: every worker must advertise the IP its peers can reach
# it on, not a loopback. Only unset or "auto" reach here (a concrete value such
# as the local test's 127.0.0.1 is left untouched), and both mean "resolve a
# routable IP for me". If detection fails we die rather than fall back to
# Mooncake auto-detect, which may pick a loopback/docker NIC and break discovery
# with a far more obscure error later.
if [[ -z "${MC_TCP_BIND_ADDRESS:-}" || "${MC_TCP_BIND_ADDRESS}" == "auto" ]]; then
  detected_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -z "${detected_ip}" ]]; then
    die "could not detect a routable host IP (hostname -I empty) on $(hostname -s); set MC_TCP_BIND_ADDRESS or MIGRATION_NIC explicitly"
  fi
  export MC_TCP_BIND_ADDRESS="${detected_ip}"
fi

# Divide the global [start, end) layer span into `count` contiguous parts and
# echo "<sliceStart> <sliceEnd>" for part `index` (0-based). The remainder goes
# to the lowest-indexed parts, so the slices tile the range exactly — no gaps
# (which would hang a request) and no overlaps (which would double-ack).
computeLayerSlice() {
  local start="$1" end="$2" count="$3" index="$4"
  local total=$(( end - start ))
  local base=$(( total / count ))
  local rem=$(( total % count ))
  local off size
  if (( index < rem )); then
    off=$(( index * (base + 1) ))
    size=$(( base + 1 ))
  else
    off=$(( rem * (base + 1) + (index - rem) * base ))
    size="${base}"
  fi
  echo "$(( start + off )) $(( start + off + size ))"
}

# Resolve (role, role-local index) from rank under whichever launch mode is
# active, then defer the topology computation to the shared block below.
extra_args=()
case "${WORKER_ROLE}" in
  prefill)
    if (( rank >= NUM_PREFILL )); then
      die "WORKER_ROLE=prefill rank ${rank} out of range; expected -np ${NUM_PREFILL}."
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
      die "WORKER_ROLE=decode rank ${rank} out of range; expected -np ${NUM_DECODE}."
    fi
    role="decode"
    role_index="${rank}"
    extra_args+=(--no-kafka)
    ;;
  "")
    # Single-launch mode (existing MooncakeMpiDiscovery).
    if (( rank >= TOTAL_WORKERS )); then
      die "rank ${rank} out of range; NUM_PREFILL=${NUM_PREFILL} + NUM_DECODE=${NUM_DECODE} = ${TOTAL_WORKERS} workers. Launch mpirun with -np ${TOTAL_WORKERS}."
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
    die "unknown WORKER_ROLE='${WORKER_ROLE}' (expected: prefill, decode, or unset)"
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

# Per-rank KV layer slice. When a global span is configured, split it into one
# contiguous slice per worker of this role (NUM_PREFILL slices for prefill,
# NUM_DECODE for decode) and hand this worker its role_index'th slice, in order.
# Unset/zero LAYER_END leaves the worker layer-agnostic (owns all layers).
layer_args=()
if [[ -n "${LAYER_END:-}" && "${LAYER_END}" != "0" ]]; then
  layer_start_global="${LAYER_START:-0}"
  if (( LAYER_END <= layer_start_global )); then
    die "LAYER_END (${LAYER_END}) must exceed LAYER_START (${layer_start_global})"
  fi
  if [[ "${role}" == "prefill" ]]; then shard_count="${NUM_PREFILL}"; else shard_count="${NUM_DECODE}"; fi
  if (( LAYER_END - layer_start_global < shard_count )); then
    die "cannot shard $(( LAYER_END - layer_start_global )) layer(s) across ${shard_count} ${role} worker(s)"
  fi
  read -r slice_start slice_end \
    < <(computeLayerSlice "${layer_start_global}" "${LAYER_END}" "${shard_count}" "${role_index}")
  layer_args+=(--layer-start "${slice_start}" --layer-end "${slice_end}")
fi

# ${peer_args[@]+...} keeps `set -u` happy when a prefill ends up with no peers.
exec "${WORKER_BIN}" \
  --metadata "${METADATA}" \
  --name "${name}" \
  ${peer_args[@]+"${peer_args[@]}"} \
  --host-dram-bytes "${HOST_DRAM_BYTES}" \
  --discovery-timeout-sec "${DISCOVERY_TIMEOUT_SEC}" \
  ${layer_args[@]+"${layer_args[@]}"} \
  ${extra_args[@]+"${extra_args[@]}"}
