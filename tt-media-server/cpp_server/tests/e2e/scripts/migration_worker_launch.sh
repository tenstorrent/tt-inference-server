#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Per-worker launcher for mooncake_kv_migration_worker (the REAL KV data-plane
# worker that supersedes bringup_mooncake_worker). deploy_migration_workers.sh
# sets the env below and runs exactly one of these per host, so a relaunch
# reproduces the worker identically.
#
# It only translates the deploy's env into the worker's flags — all topology
# (how many prefill/decode, which physical host) is decided by the deploy. We
# pass --role explicitly (from WORKER_ROLE): the worker can infer role from a
# prefill*/decode* --name prefix, but real table tags (bh-glx-…, host-…) don't
# match that prefix, so relying on inference makes the worker abort.
#
# Contract: WORKER_TAG is this worker's logical identity and is used verbatim as
#   --name (the Mooncake/rpc_meta discovery key) and --host (the table's
#   fabric_node_host this worker owns). A peer is referenced by the SAME tag: a
#   PEERS entry "decode-3" resolves the worker whose WORKER_TAG is "decode-3".
# These must be the same string for discovery + routing to line up, so the
# deploy pins them to a single tag per worker.
#
# Peers (PEERS): CSV of peer tags for THIS worker, role-agnostic — the deploy
#   fills it (default: prefill → every decode for control TABLE_EXCHANGE +
#   migrate; decode → none) and this launcher just forwards it. The worker
#   resolves each tag for control host:port from metadata; role decides who
#   initiates (prefill = sender).
#
# Required env (all roles): WORKER_ROLE, WORKER_BIN, METADATA, WORKER_TAG,
#   HEALTH_PORT.
# Decode also needs DECODE_TABLE. Prefill needs PREFILL_TABLE and non-empty
#   PEERS; DECODE_TABLE is an optional disk fallback to control TABLE_EXCHANGE.
# Optional: PEERS (see above), KAFKA_BROKERS, KAFKA_GROUP_ID,
#   KV_MIGRATION_MODE (device|dry-run; applies to both roles),
#   PROTOCOL (tcp|rdma; Mooncake wire, applies to both roles; unset => the
#     worker's default tcp / its own MIGRATION_MOONCAKE_PROTOCOL env),
#   RDMA_NICS (CSV of RDMA NIC device names, e.g. "mlx5_0,mlx5_1"; rdma only;
#     unset => auto-discover the present NIC),
#   MC_TCP_BIND_ADDRESS
#   (unset/"auto" => detect this host's routable IP so peers can reach it),
#   CONTROL_PORT (KV control port a decode binds + publishes to metadata;
#   defaults to the worker's own default when unset), DEVICE_MAP (legacy file
#   fallback), ENGINE_HANDOFF_PORT (listen for DeviceMap over localhost; preferred
#   when deploy pushes via engine_handoff_sender). Omit both for discovery-only.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 2; }

: "${WORKER_ROLE:?WORKER_ROLE required}"
: "${WORKER_BIN:?WORKER_BIN required}"
: "${METADATA:?METADATA required}"
: "${WORKER_TAG:?WORKER_TAG required}"
: "${HEALTH_PORT:?HEALTH_PORT required}"

# Cross-host deployment: advertise the IP peers can reach us on, never a
# loopback. Only unset/"auto" is resolved here; a concrete value (e.g. a local
# test's 127.0.0.1) is left untouched. Fail loud rather than let Mooncake
# auto-detect pick a docker/loopback NIC and break discovery obscurely later.
if [[ -z "${MC_TCP_BIND_ADDRESS:-}" || "${MC_TCP_BIND_ADDRESS}" == "auto" ]]; then
  detected_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  [[ -n "${detected_ip}" ]] || \
    die "could not detect a routable host IP (hostname -I empty) on $(hostname -s); set MC_TCP_BIND_ADDRESS"
  export MC_TCP_BIND_ADDRESS="${detected_ip}"
fi

health_args=()
[[ "${HEALTH_PORT}" != "0" ]] && health_args+=(--health-port "${HEALTH_PORT}")
# KV control port, fleet-wide from the deploy config. Passed explicitly rather
# than relying on the worker's built-in default so the port is configured in one
# place. Decode binds + publishes it; prefill only uses it as a fallback for a
# peer that hasn't published its control endpoint to metadata.
control_args=()
[[ -n "${CONTROL_PORT:-}" ]] && control_args=(--control-port "${CONTROL_PORT}")
peer_control_args=()
[[ -n "${CONTROL_PORT:-}" ]] && peer_control_args=(--peer-control-port "${CONTROL_PORT}")
# Transfer-plane: socket DeviceMap (deploy pushes after start) wins over file.
device_args=()
if [[ -n "${ENGINE_HANDOFF_PORT:-}" && "${ENGINE_HANDOFF_PORT}" != "0" ]]; then
  device_args=(--engine-handoff-port "${ENGINE_HANDOFF_PORT}")
elif [[ -n "${DEVICE_MAP:-}" ]]; then
  device_args=(--device-map "${DEVICE_MAP}")
fi

# Peers are a GENERIC per-worker input (PEERS = CSV of peer tags), independent of
# role: a worker is just a migration worker with a peer list, resolved through
# the metadata service. The deploy decides the set (default: prefill gets every
# decode; a decode gets none), so this launcher never branches peers on role —
# it just forwards whatever it was handed. Each peer's control host:port is
# discovered from its published "kv_control/<tag>"; CONTROL_PORT is only the
# fallback for a peer that hasn't published yet.
peer_args=()
IFS=',' read -ra peer_tags <<<"${PEERS:-}"
for tag in "${peer_tags[@]}"; do
  [[ -n "${tag}" ]] && peer_args+=(--peer "${tag}")
done

# Mooncake wire transport, applies to both roles. Only forwarded when PROTOCOL
# is set, so the worker's tcp default (and its MIGRATION_MOONCAKE_PROTOCOL env
# fallback) stay intact when the deploy doesn't pin it. --protocol wins over the
# env in the worker, so a set PROTOCOL is authoritative.
protocol_args=()
[[ -n "${PROTOCOL:-}" ]] && protocol_args=(--protocol "${PROTOCOL}")
# RDMA NIC allowlist (rdma only): CSV of device names -> one --rdma-nic each.
# Unset/empty => the worker auto-discovers the single present NIC.
rdma_nic_args=()
IFS=',' read -ra rdma_nics <<<"${RDMA_NICS:-}"
for nic in "${rdma_nics[@]}"; do
  [[ -n "${nic}" ]] && rdma_nic_args+=(--rdma-nic "${nic}")
done

case "${WORKER_ROLE}" in
  decode)
    : "${DECODE_TABLE:?DECODE_TABLE required for decode}"
    # Receiver: registers the bounce buffer + serves control (TABLE_EXCHANGE reply +
    # migrate). Prefill initiates; decode does not. No Kafka.
    exec "${WORKER_BIN}" \
      --role "${WORKER_ROLE}" \
      --metadata "${METADATA}" --name "${WORKER_TAG}" --host "${WORKER_TAG}" \
      --table "${DECODE_TABLE}" \
      ${peer_args[@]+"${peer_args[@]}"} \
      ${peer_control_args[@]+"${peer_control_args[@]}"} \
      ${control_args[@]+"${control_args[@]}"} \
      ${device_args[@]+"${device_args[@]}"} \
      ${protocol_args[@]+"${protocol_args[@]}"} \
      ${rdma_nic_args[@]+"${rdma_nic_args[@]}"} \
      ${health_args[@]+"${health_args[@]}"}
    ;;
  prefill)
    : "${PREFILL_TABLE:?PREFILL_TABLE required for prefill}"
    decode_table_args=()
    [[ -n "${DECODE_TABLE:-}" ]] && decode_table_args=(--decode-table "${DECODE_TABLE}")
    # The sender needs a control channel to every peer it might route to.
    (( ${#peer_args[@]} > 0 )) || die "prefill has no peers (PEERS empty)"
    # One consumer group per prefill so every prefill sees each request (Kafka
    # delivers a record to one member per group => broadcast across prefills).
    export KAFKA_GROUP_ID="${KAFKA_GROUP_ID:-migration-workers-prefill-${WORKER_TAG}}"
    exec "${WORKER_BIN}" \
      --role "${WORKER_ROLE}" \
      --metadata "${METADATA}" --name "${WORKER_TAG}" --host "${WORKER_TAG}" \
      --prefill-table "${PREFILL_TABLE}" \
      ${decode_table_args[@]+"${decode_table_args[@]}"} \
      "${peer_args[@]}" \
      ${peer_control_args[@]+"${peer_control_args[@]}"} \
      ${device_args[@]+"${device_args[@]}"} \
      ${protocol_args[@]+"${protocol_args[@]}"} \
      ${rdma_nic_args[@]+"${rdma_nic_args[@]}"} \
      ${health_args[@]+"${health_args[@]}"}
    ;;
  *)
    die "WORKER_ROLE must be prefill|decode (got '${WORKER_ROLE}')"
    ;;
esac
