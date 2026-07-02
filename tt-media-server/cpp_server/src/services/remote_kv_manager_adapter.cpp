// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/remote_kv_manager_adapter.hpp"

#include <algorithm>

#include "utils/logger.hpp"

namespace tt::services {

RemoteKVManagerAdapter::RemoteKVManagerAdapter(
    std::unique_ptr<IRemoteKVManager> kvManager, uint32_t layersPerChunk)
    : kvManager_(std::move(kvManager)), layersPerChunk_(layersPerChunk) {
  if (!kvManager_) {
    TT_LOG_ERROR("[RemoteKVManagerAdapter] null kvManager");
  }
}

RemoteKVManagerAdapter::BurstId RemoteKVManagerAdapter::start_burst(
    MigrationToken uuid) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (bursts_.count(uuid)) {
    TT_LOG_WARN("[RemoteKVManagerAdapter] burst {} already open", uuid);
    return uuid;
  }

  bursts_.emplace(uuid, BurstState{});
  TT_LOG_DEBUG("[RemoteKVManagerAdapter] started burst {}", uuid);
  return uuid;
}

void RemoteKVManagerAdapter::enqueue_migration_in_burst(
    BurstId burst, int /*remote_endpoint_id*/, uint32_t src_slot,
    uint32_t dst_slot, uint32_t layer_start, uint32_t layer_end_exclusive,
    uint32_t pos_start, uint32_t pos_end_exclusive) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto it = bursts_.find(burst);
  if (it == bursts_.end()) {
    TT_LOG_ERROR(
        "[RemoteKVManagerAdapter] enqueue_migration_in_burst: burst {} not "
        "found",
        burst);
    return;
  }

  if (it->second.finished) {
    TT_LOG_ERROR(
        "[RemoteKVManagerAdapter] enqueue_migration_in_burst: burst {} already "
        "finished",
        burst);
    return;
  }

  MigrationRequest request{
      .src_slot = src_slot,
      .dst_slot = dst_slot,
      .layer_begin = layer_start,
      .layer_end = layer_end_exclusive,
      .src_position_begin = pos_start,
      .src_position_end = pos_end_exclusive,
      .dst_position_begin = pos_start,
      .dst_position_end = pos_end_exclusive,
  };

  uint64_t migrationId = kvManager_->migrate(request);
  it->second.migrationIds.push_back(migrationId);

  inFlight_.emplace(
      migrationId,
      InFlightMigration{.token = burst, .isBurstMember = true, .burstId = burst});

  TT_LOG_DEBUG(
      "[RemoteKVManagerAdapter] enqueued migration {} in burst {} "
      "(slot {}→{}, layers [{},{}), pos [{},{}))",
      migrationId, burst, src_slot, dst_slot, layer_start, layer_end_exclusive,
      pos_start, pos_end_exclusive);
}

MigrationToken RemoteKVManagerAdapter::finish_burst(BurstId burst) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto it = bursts_.find(burst);
  if (it == bursts_.end()) {
    TT_LOG_ERROR("[RemoteKVManagerAdapter] finish_burst: burst {} not found",
                 burst);
    return burst;
  }

  it->second.finished = true;
  TT_LOG_DEBUG("[RemoteKVManagerAdapter] finished burst {} with {} migrations",
               burst, it->second.migrationIds.size());
  return burst;
}

MigrationToken RemoteKVManagerAdapter::migrate(
    int /*remote_endpoint_id*/, uint32_t src_slot, uint32_t dst_slot,
    uint32_t layer_start, uint32_t layer_end_exclusive, uint32_t pos_start,
    uint32_t pos_end_exclusive) {
  MigrationRequest request{
      .src_slot = src_slot,
      .dst_slot = dst_slot,
      .layer_begin = layer_start,
      .layer_end = layer_end_exclusive,
      .src_position_begin = pos_start,
      .src_position_end = pos_end_exclusive,
      .dst_position_begin = pos_start,
      .dst_position_end = pos_end_exclusive,
  };

  uint64_t migrationId = kvManager_->migrate(request);
  MigrationToken token = nextToken_.fetch_add(1, std::memory_order_relaxed);

  {
    std::lock_guard<std::mutex> lock(mtx_);
    inFlight_.emplace(migrationId, InFlightMigration{.token = token,
                                                     .isBurstMember = false,
                                                     .burstId = 0});
  }

  TT_LOG_DEBUG(
      "[RemoteKVManagerAdapter] migrate: token={}, migrationId={}, "
      "slot {}→{}, layers [{},{}), pos [{},{}))",
      token, migrationId, src_slot, dst_slot, layer_start, layer_end_exclusive,
      pos_start, pos_end_exclusive);

  return token;
}

int RemoteKVManagerAdapter::poll() {
  std::vector<uint64_t> toCheck;
  std::vector<BurstId> completedBursts;

  {
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& [migrationId, info] : inFlight_) {
      toCheck.push_back(migrationId);
    }
  }

  int completions = 0;

  for (uint64_t migrationId : toCheck) {
    MigrationStatus status = kvManager_->getStatus(migrationId);

    if (status == MigrationStatus::IN_PROGRESS ||
        status == MigrationStatus::UNKNOWN) {
      continue;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = inFlight_.find(migrationId);
    if (it == inFlight_.end()) continue;

    InFlightMigration info = it->second;
    inFlight_.erase(it);
    ++completions;

    if (info.isBurstMember) {
      auto burstIt = bursts_.find(info.burstId);
      if (burstIt != bursts_.end()) {
        auto& ids = burstIt->second.migrationIds;
        ids.erase(std::remove(ids.begin(), ids.end(), migrationId), ids.end());

        if (status == MigrationStatus::FAILED) {
          completedBursts.push_back(info.burstId);
          TT_LOG_WARN(
              "[RemoteKVManagerAdapter] migration {} in burst {} failed",
              migrationId, info.burstId);
        } else if (burstIt->second.finished && ids.empty()) {
          completedBursts.push_back(info.burstId);
        }
      }
    } else {
      if (onComplete_) {
        MigrationCompleteEvent event{
            .token = info.token,
            .ok = (status == MigrationStatus::SUCCESSFUL),
        };
        onComplete_(event);
      }
    }
  }

  for (BurstId burstId : completedBursts) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = bursts_.find(burstId);
    if (it == bursts_.end()) continue;

    bool ok = it->second.migrationIds.empty();
    bursts_.erase(it);

    if (onComplete_) {
      MigrationCompleteEvent event{
          .token = burstId,
          .ok = ok,
      };
      onComplete_(event);
    }
  }

  return completions;
}

void RemoteKVManagerAdapter::on_migration_complete(
    std::function<void(const MigrationCompleteEvent&)> cb) {
  onComplete_ = std::move(cb);
}

void RemoteKVManagerAdapter::on_migration_received(
    std::function<void(const MigrationReceivedEvent&)> cb) {
  onReceived_ = std::move(cb);
}

void RemoteKVManagerAdapter::on_migration_failed(
    std::function<void(const MigrationFailedEvent&)> cb) {
  onFailed_ = std::move(cb);
}

void RemoteKVManagerAdapter::on_endpoint_disconnected(
    std::function<void(const EndpointDisconnectedEvent&)> cb) {
  onDisconnected_ = std::move(cb);
}

void RemoteKVManagerAdapter::on_connection_received(
    std::function<void(int remote_endpoint_id)> cb) {
  onConnectionReceived_ = std::move(cb);
}

void RemoteKVManagerAdapter::connect_to(int /*remote_endpoint_id*/,
                                        const std::string& /*role*/,
                                        const std::string& /*service_name*/) {
  // Mooncake discovery handles connection; no-op here
}

void RemoteKVManagerAdapter::wait_ready(int /*timeout_ms*/) {
  // Kafka-based system is ready when producer/consumer are connected
  // (handled at construction time)
}

void RemoteKVManagerAdapter::shutdown(bool /*drain*/) {
  // IRemoteKVManager handles cleanup in destructor
}

}  // namespace tt::services
