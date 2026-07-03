// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/remote_kv_manager_adapter.hpp"

#include <stdexcept>

#include "utils/logger.hpp"

namespace tt::services {

RemoteKVManagerAdapter::RemoteKVManagerAdapter(
    std::unique_ptr<IRemoteKVManager> kvManager,
    std::chrono::milliseconds shutdownTimeout)
    : kvManager_(std::move(kvManager)), shutdownTimeout_(shutdownTimeout) {
  if (!kvManager_) {
    TT_LOG_ERROR("[RemoteKVManagerAdapter] null kvManager");
  }
}

RemoteKVManagerAdapter::BurstId RemoteKVManagerAdapter::start_burst(
    MigrationToken /*uuid*/) {
  throw std::runtime_error(
      "[RemoteKVManagerAdapter] burst methods not supported - use "
      "MigrationLayerClientAdapter for burst-based migrations");
}

void RemoteKVManagerAdapter::enqueue_migration_in_burst(
    BurstId /*burst*/, int /*remote_endpoint_id*/, uint32_t /*src_slot*/,
    uint32_t /*dst_slot*/, uint32_t /*layer_start*/,
    uint32_t /*layer_end_exclusive*/, uint32_t /*pos_start*/,
    uint32_t /*pos_end_exclusive*/) {
  throw std::runtime_error(
      "[RemoteKVManagerAdapter] burst methods not supported - use "
      "MigrationLayerClientAdapter for burst-based migrations");
}

RemoteKVManagerAdapter::MigrationToken RemoteKVManagerAdapter::finish_burst(
    BurstId /*burst*/) {
  throw std::runtime_error(
      "[RemoteKVManagerAdapter] burst methods not supported - use "
      "MigrationLayerClientAdapter for burst-based migrations");
}

RemoteKVManagerAdapter::MigrationToken RemoteKVManagerAdapter::migrate(
    int /*remote_endpoint_id*/, uint32_t src_slot, uint32_t dst_slot,
    uint32_t layer_start, uint32_t layer_end_exclusive, uint32_t pos_start,
    uint32_t pos_end_exclusive) {
  // remote_endpoint_id ignored: Kafka topic routing is not endpoint-specific.
  MigrationRequest request{
      .src_slot = src_slot,
      .dst_slot = dst_slot,
      .layer_begin = layer_start,
      .layer_end = layer_end_exclusive,
      // ALLOCATE prefix copies: src and dst position ranges are identical.
      .src_position_begin = pos_start,
      .src_position_end = pos_end_exclusive,
      .dst_position_begin = pos_start,
      .dst_position_end = pos_end_exclusive,
  };

  std::lock_guard<std::mutex> lock(mtx_);
  if (shutdownRequested_) {
    throw std::runtime_error(
        "[RemoteKVManagerAdapter] migrate() called after shutdown()");
  }
  uint64_t migrationId = kvManager_->migrate(request);
  inFlight_.emplace(migrationId, InFlightMigration{.token = migrationId});

  TT_LOG_DEBUG(
      "[RemoteKVManagerAdapter] migrate: migrationId={}, "
      "slot {}->{}, layers [{},{}), pos [{},{}))",
      migrationId, src_slot, dst_slot, layer_start, layer_end_exclusive,
      pos_start, pos_end_exclusive);

  return migrationId;
}

int RemoteKVManagerAdapter::poll() {
  std::vector<uint64_t> toCheck;

  {
    std::lock_guard<std::mutex> lock(mtx_);
    toCheck.reserve(inFlight_.size());
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

    MigrationToken token;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      auto it = inFlight_.find(migrationId);
      if (it == inFlight_.end()) continue;

      token = it->second.token;
      inFlight_.erase(it);
    }
    drainCv_.notify_all();
    ++completions;

    // MigrationComplete and MigrationFailed are mutually exclusive per the
    // interface contract. Fire onFailed_ for failures, onComplete_ for success.
    // Fallback: if onFailed_ is not registered, fire onComplete_ with ok=false.
    if (status == MigrationStatus::SUCCESSFUL) {
      if (onComplete_) {
        onComplete_({.token = token, .ok = true});
      }
    } else {
      // TODO(reason): IRemoteKVManager only reports FAILED with no context.
      // Add getFailureInfo(id) to expose failure details for retry/quarantine.
      if (onFailed_) {
        onFailed_({.token = token, .remote_endpoint_id = -1, .reason = 0});
      } else if (onComplete_) {
        onComplete_({.token = token, .ok = false});
      }
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
  // (handled at construction time in RemoteKVManagerImpl)
}

void RemoteKVManagerAdapter::shutdown(bool drain) {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (shutdownRequested_) {
      return;
    }
    shutdownRequested_ = true;
  }

  if (!drain) {
    return;
  }

  std::unique_lock<std::mutex> lock(mtx_);
  const auto deadline = std::chrono::steady_clock::now() + shutdownTimeout_;

  while (!inFlight_.empty()) {
    if (drainCv_.wait_until(lock, deadline) == std::cv_status::timeout) {
      TT_LOG_WARN(
          "[RemoteKVManagerAdapter] shutdown timeout with {} migrations "
          "still in flight",
          inFlight_.size());
      break;
    }
  }
}

}  // namespace tt::services
