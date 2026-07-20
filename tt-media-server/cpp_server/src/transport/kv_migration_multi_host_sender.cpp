// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_migration_multi_host_sender.hpp"

#include <utility>
#include <vector>

#include "transport/kv_migration_orchestrator.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

KvMigrationMultiHostSender::KvMigrationMultiHostSender(
    std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
    std::shared_ptr<const IKvTable> prefillTable,
    std::shared_ptr<const IKvTable> decodeTable, std::string prefillHost,
    std::unordered_map<std::string, std::shared_ptr<KvControlChannel>> channels,
    DeviceMapFn deviceMap, WorkerHealth* health)
    : engine_(std::move(engine)),
      device_(device),
      prefill_table_(std::move(prefillTable)),
      decode_table_(std::move(decodeTable)),
      prefill_host_(std::move(prefillHost)),
      health_(health),
      channels_(std::move(channels)) {
  // One staging pool shared by every per-host bounce sender (serial fan-out),
  // double-pinned when a device map is supplied (DRISC).
  staging_ = std::make_shared<KvStagingPool>(engine_, defaultStagingBytes(),
                                             std::move(deviceMap));
  for (const auto& [host, channel] : channels_) {
    senders_[host] = std::make_unique<MooncakeKvSender>(
        engine_, device_, prefill_table_, decode_table_, prefill_host_, host,
        staging_, health_);
  }
}

bool KvMigrationMultiHostSender::addHost(
    const std::string& host, std::shared_ptr<KvControlChannel> channel) {
  if (!channel) {
    TT_LOG_ERROR(
        "[KvMigrationMultiHostSender] addHost('{}'): null control channel",
        host);
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const bool isNew = senders_.count(host) == 0;
  channels_[host] = std::move(channel);
  if (isNew) {
    senders_[host] = std::make_unique<MooncakeKvSender>(
        engine_, device_, prefill_table_, decode_table_, prefill_host_, host,
        staging_, health_);
    TT_LOG_INFO(
        "[KvMigrationMultiHostSender] added late decode host '{}' "
        "(hosts={})",
        host, senders_.size());
  }
  return true;
}

std::size_t KvMigrationMultiHostSender::hostCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return senders_.size();
}

bool KvMigrationMultiHostSender::migrate(uint64_t uuid,
                                         const MigrationRequest& request) {
  if (!decode_table_) {
    TT_LOG_ERROR(
        "[KvMigrationMultiHostSender] migrate(uuid={}): no decode "
        "table",
        uuid);
    return false;
  }
  // Routing: the exact decode hosts this request lands on, sorted so the
  // fan-out order is deterministic.
  const std::vector<std::string> hosts =
      hostsForRequest(*decode_table_, request.dstSlice());
  if (hosts.empty()) {
    TT_LOG_ERROR(
        "[KvMigrationMultiHostSender] migrate(uuid={}): request touches no "
        "decode hosts",
        uuid);
    return false;
  }

  struct HostLeg {
    std::string host;
    std::shared_ptr<KvControlChannel> channel;
    MooncakeKvSender* sender = nullptr;
  };
  // Snapshot host -> channel/sender under the lock so a concurrent addHost()
  // cannot invalidate the map mid-fan-out; the shared_ptr keeps each channel
  // alive even if the connector replaces it during this migrate().
  std::vector<HostLeg> legs;
  legs.reserve(hosts.size());
  bool ok = true;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const std::string& host : hosts) {
      const auto chIt = channels_.find(host);
      const auto sIt = senders_.find(host);
      if (chIt == channels_.end() || !chIt->second || sIt == senders_.end()) {
        TT_LOG_ERROR(
            "[KvMigrationMultiHostSender] migrate(uuid={}): no control "
            "channel for decode host '{}' (resolution missing)",
            uuid, host);
        ok = false;
        continue;
      }
      legs.push_back(HostLeg{host, chIt->second, sIt->second.get()});
    }
  }

  for (const HostLeg& leg : legs) {
    KvMigrationSender perHost(*leg.channel, *leg.sender);
    if (!perHost.migrate(uuid, request)) {
      TT_LOG_ERROR(
          "[KvMigrationMultiHostSender] migrate(uuid={}): decode host '{}' "
          "failed",
          uuid, leg.host);
      ok = false;
    }
  }
  return ok;
}

}  // namespace tt::transport
