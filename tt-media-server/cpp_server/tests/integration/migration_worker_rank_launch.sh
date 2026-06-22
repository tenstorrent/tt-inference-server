#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Per-rank launcher for the #4294 MPI discovery test. mpirun starts one copy of
# this per rank; we turn the rank into a worker identity + its peer set and exec
# bringup_mooncake_worker. The mapping mirrors a disaggregated topology:
#
#   ranks 0..3   -> prefill-0..3   (4 prefill workers)
#   ranks 4..19  -> decode-0..15   (16 decode workers)
#
#   prefill-p peers = decode-(4p .. 4p+3)   (each prefill fans out to 4 decode)
#   decode-d  peer  = prefill-(d / 4)        (symmetric: decode points back)
#
# Symmetric peering means every worker has at least one peer and only reaches
# READY once it has resolved all of them — so 20 "CONNECTED" lines proves the
# whole mesh wired up. Required env: WORKER_BIN, METADATA, HOST_DRAM_BYTES,
# DISCOVERY_TIMEOUT_SEC.
set -euo pipefail

readonly NUM_PREFILL=4
readonly DECODE_PER_PREFILL=4  # 16 decode / 4 prefill

rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}"
if [[ -z "${rank}" ]]; then
  echo "ERROR: no MPI rank in environment (OMPI_COMM_WORLD_RANK/PMI_RANK)" >&2
  exit 2
fi

: "${WORKER_BIN:?WORKER_BIN must point at bringup_mooncake_worker}"
: "${METADATA:?METADATA discovery URI required}"
: "${HOST_DRAM_BYTES:?HOST_DRAM_BYTES required}"
: "${DISCOVERY_TIMEOUT_SEC:?DISCOVERY_TIMEOUT_SEC required}"

peer_args=()
if (( rank < NUM_PREFILL )); then
  prefill_index="${rank}"
  name="prefill-${prefill_index}"
  base=$(( prefill_index * DECODE_PER_PREFILL ))
  for offset in $(seq 0 $(( DECODE_PER_PREFILL - 1 ))); do
    peer_args+=(--peer "decode-$(( base + offset ))")
  done
else
  decode_index=$(( rank - NUM_PREFILL ))
  prefill_index=$(( decode_index / DECODE_PER_PREFILL ))
  name="decode-${decode_index}"
  peer_args+=(--peer "prefill-${prefill_index}")
fi

exec "${WORKER_BIN}" \
  --metadata "${METADATA}" \
  --name "${name}" \
  "${peer_args[@]}" \
  --host-dram-bytes "${HOST_DRAM_BYTES}" \
  --discovery-timeout-sec "${DISCOVERY_TIMEOUT_SEC}"
