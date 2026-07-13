// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_migration_multi_host_sender.hpp"

#include <utility>

#include "transport/kv_migration_orchestrator.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

KvMigrationMultiHostSender::KvMigrationMultiHostSender(
    std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
    std::shared_ptr<const IKvTable> prefillTable,
    std::shared_ptr<const IKvTable> decodeTable, std::string prefillHost,
    std::unordered_map<std::string, KvControlChannel*> channels,
    WorkerHealth* health)
    : decode_table_(std::move(decodeTable)), channels_(std::move(channels)) {
  // One staging pool, registered once, shared by every per-host sender (safe
  // because the fan-out in migrate() is serial).
  staging_ = std::make_shared<KvStagingPool>(engine);
  // One per-host sender, built once: each holds the destination addressing for
  // its decode host (whole-table-stable), reused across migrations.
  for (const auto& [host, channel] : channels_) {
    senders_[host] = std::make_unique<MooncakeKvSender>(
        engine, device, prefillTable, decode_table_, prefillHost, host,
        staging_, health);
  }
}

bool KvMigrationMultiHostSender::migrate(uint64_t uuid,
                                         const MigrationRequest& request) {
  if (!decode_table_) {
    TT_LOG_ERROR(
        "[KvMigrationMultiHostSender] migrate(uuid={}): no decode table", uuid);
    return false;
  }
  // Routing: the exact set of decode hosts this request lands on (sorted, so
  // the fan-out order is deterministic).
  const std::vector<std::string> hosts =
      hostsForRequest(*decode_table_, request.dstSlice());
  if (hosts.empty()) {
    TT_LOG_ERROR(
        "[KvMigrationMultiHostSender] migrate(uuid={}): request touches no "
        "decode "
        "hosts",
        uuid);
    return false;
  }

  bool ok = true;
  for (const std::string& host : hosts) {
    const auto chIt = channels_.find(host);
    const auto sIt = senders_.find(host);
    if (chIt == channels_.end() || sIt == senders_.end()) {
      TT_LOG_ERROR(
          "[KvMigrationMultiHostSender] migrate(uuid={}): no control channel "
          "for "
          "decode host '{}' (resolution missing)",
          uuid, host);
      ok = false;
      continue;  // attempt the rest so the report is comprehensive
    }
    // Reuse the single-host orchestrator for this host's Begin/MirrorReady/
    // Done/Ack sequence over its own channel.
    KvMigrationSender perHost(*chIt->second, *sIt->second);
    if (!perHost.migrate(uuid, request)) {
      TT_LOG_ERROR(
          "[KvMigrationMultiHostSender] migrate(uuid={}): decode host '{}' "
          "failed",
          uuid, host);
      ok = false;
    }
  }
  return ok;
}

}  // namespace tt::transport
