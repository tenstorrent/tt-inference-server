// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "transport/i_device_io.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/kv_control_channel.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/kv_table_view.hpp"
#include "transport/mooncake_kv_sender.hpp"

namespace tt::transport {

class WorkerHealth;

/**
 * @brief Sender-side fan-out across the N decode hosts a slot spans (n->m).
 *
 * A whole-slot migration is spread across several decode hosts (layers on
 * different meshes, replicas in different device groups). Each decode host is
 * its own receiver process with its own Mooncake segment + mirror + control
 * channel. This composes one per-host MooncakeKvSender (the destination
 * addressing for that host, built from the shared decode table) and reuses
 * KvMigrationSender for the per-host Begin/MirrorReady/Done/Ack protocol.
 *
 * Two concerns are kept separate:
 *   - ROUTING — which hosts a request touches — is computed from the decode
 *     table (hostsForRequest), so it is exact and table-driven.
 *   - RESOLUTION — host -> control channel — is injected. A static map drives
 *     standalone tests and the e2e; a discovery service supplies the same map
 *     in production, with no change to this class.
 *
 * The injected map defines the known decode cluster; each migrate() drives only
 * the subset of hosts the request actually touches. Owns no threads; the
 * per-host receivers run in their own processes.
 */
class KvMigrationMultiHostSender {
 public:
  /// @param channels host -> already-connected control channel for that decode
  ///        host's receiver. One per-host MooncakeKvSender is built up front
  ///        for each key (destination layout is whole-table-stable, so it is
  ///        reused across migrations).
  /// @param health optional; forwarded to each per-host sender so a stale-peer
  ///        transfer failure bumps the re-resolve counters (observability
  ///        only).
  KvMigrationMultiHostSender(
      std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
      std::shared_ptr<const IKvTable> prefillTable,
      std::shared_ptr<const IKvTable> decodeTable, std::string prefillHost,
      std::unordered_map<std::string, KvControlChannel*> channels,
      WorkerHealth* health = nullptr);

  /**
   * @brief Drive the migration to every decode host the request touches.
   *
   * Hosts are driven in a deterministic (sorted) order. A host involved in the
   * request but absent from the channel map, or whose per-host migration fails,
   * makes the whole call fail — but the remaining hosts are still attempted so
   * the failure is reported comprehensively.
   *
   * @return true iff every involved host completed. false means the slot is
   * only partially migrated across the cluster; the same all-or-nothing retry
   * contract as KvMigrationSender::migrate applies (re-drive the same request).
   */
  bool migrate(uint64_t uuid, const MigrationRequest& request);

  /// Number of known decode hosts (channel-map size).
  std::size_t hostCount() const { return senders_.size(); }

 private:
  std::shared_ptr<const IKvTable> decode_table_;
  std::unordered_map<std::string, KvControlChannel*> channels_;
  std::unordered_map<std::string, std::unique_ptr<MooncakeKvSender>> senders_;
};

}  // namespace tt::transport
