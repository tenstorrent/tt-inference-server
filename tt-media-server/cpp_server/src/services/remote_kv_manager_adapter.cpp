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
    MigrationToken uuid) {
  // BurstId == uuid (caller-supplied). Insert an OPEN group; finish_burst()
  // flips `closed` and poll() fires the terminal event only when both
  // `closed && pendingKafkaIds.empty()`.
  std::lock_guard<std::mutex> lock(mtx_);
  if (shutdownRequested_) {
    throw std::runtime_error(
        "[RemoteKVManagerAdapter] start_burst() called after shutdown()");
  }

  auto [it, inserted] =
      groups_.emplace(uuid, MigrationGroup{.token = uuid,
                                           .pendingKafkaIds = {},
                                           .closed = false,
                                           .failed = false,
                                           .failedReported = false});
  if (!inserted) {
    // Duplicate uuid in flight — tt-llm-engine treats this as MigrationFailed
    // {DuplicateId}; we throw so callers crash loudly at the source of the bug
    // rather than dispatching a synthetic failure event.
    throw std::runtime_error(
        "[RemoteKVManagerAdapter] start_burst(): duplicate burst uuid " +
        std::to_string(uuid) +
        " (a burst with this uuid is already in flight)");
  }

  TT_LOG_DEBUG("[RemoteKVManagerAdapter] start_burst: uuid={}", uuid);
  return uuid;
}

void RemoteKVManagerAdapter::enqueue_migration_in_burst(
    BurstId burst, int /*remote_endpoint_id*/, uint32_t src_slot,
    uint32_t dst_slot, uint32_t layer_start, uint32_t layer_end_exclusive,
    uint32_t pos_start, uint32_t pos_end_exclusive) {
  // remote_endpoint_id ignored: Kafka topic routing is not endpoint-specific.
  if (layer_end_exclusive <= layer_start) {
    throw std::invalid_argument(
        "[RemoteKVManagerAdapter] enqueue_migration_in_burst(): empty layer "
        "range");
  }
  if (pos_end_exclusive <= pos_start) {
    throw std::invalid_argument(
        "[RemoteKVManagerAdapter] enqueue_migration_in_burst(): empty position "
        "range");
  }

  std::lock_guard<std::mutex> lock(mtx_);
  if (shutdownRequested_) {
    throw std::runtime_error(
        "[RemoteKVManagerAdapter] enqueue_migration_in_burst() called after "
        "shutdown()");
  }

  auto git = groups_.find(burst);
  if (git == groups_.end()) {
    throw std::runtime_error(
        "[RemoteKVManagerAdapter] enqueue_migration_in_burst(): no burst with "
        "uuid " +
        std::to_string(burst) +
        " (did you call start_burst() first, or has finish_burst() already "
        "retired it?)");
  }
  MigrationGroup& group = git->second;
  if (group.closed) {
    throw std::runtime_error(
        "[RemoteKVManagerAdapter] enqueue_migration_in_burst(): burst uuid " +
        std::to_string(burst) + " is already closed (finish_burst was called)");
  }

  for (uint32_t layer = layer_start; layer < layer_end_exclusive; ++layer) {
    const MigrationRequest request{
        .src_slot = src_slot,
        .dst_slot = dst_slot,
        .layer_begin = layer,
        .layer_end = layer + 1,
        .src_position_begin = pos_start,
        .src_position_end = pos_end_exclusive,
        .dst_position_begin = pos_start,
        .dst_position_end = pos_end_exclusive,
    };
    const uint64_t kafkaId = kvManager_->migrate(request);
    group.pendingKafkaIds.insert(kafkaId);
    kafkaToGroup_.emplace(kafkaId, burst);
  }

  TT_LOG_DEBUG(
      "[RemoteKVManagerAdapter] enqueue_migration_in_burst: uuid={}, slot "
      "{}->{}, layers [{},{}) fanned out ({} pending Kafka req(s) total), "
      "pos [{},{})",
      burst, src_slot, dst_slot, layer_start, layer_end_exclusive,
      group.pendingKafkaIds.size(), pos_start, pos_end_exclusive);
}

RemoteKVManagerAdapter::MigrationToken RemoteKVManagerAdapter::finish_burst(
    BurstId burst) {
  // Close the group and hand the uuid back as the correlation token. Do NOT
  // fire the completion callback synchronously here — PrefillScheduler
  // emplaces `close_token` into its token_to_slot_ map AFTER finish_burst()
  // returns, and on_migration_complete_ looks that token up. Firing inside
  // finish_burst() would race the emplace and silently drop the burst.
  // poll() is the sole terminal-event dispatch point.
  std::lock_guard<std::mutex> lock(mtx_);
  auto git = groups_.find(burst);
  if (git == groups_.end()) {
    throw std::runtime_error(
        "[RemoteKVManagerAdapter] finish_burst(): no burst with uuid " +
        std::to_string(burst) +
        " (did you call start_burst() first, or is this a double-finish?)");
  }
  MigrationGroup& group = git->second;
  if (group.closed) {
    throw std::runtime_error(
        "[RemoteKVManagerAdapter] finish_burst(): burst uuid " +
        std::to_string(burst) + " is already closed (double finish_burst)");
  }
  group.closed = true;

  TT_LOG_DEBUG(
      "[RemoteKVManagerAdapter] finish_burst: uuid={}, {} Kafka req(s) "
      "still pending; poll() will fire terminal event once drained",
      burst, group.pendingKafkaIds.size());
  return burst;
}

RemoteKVManagerAdapter::MigrationToken RemoteKVManagerAdapter::migrate(
    int /*remote_endpoint_id*/, uint32_t /*src_slot*/, uint32_t /*dst_slot*/,
    uint32_t /*layer_start*/, uint32_t /*layer_end_exclusive*/,
    uint32_t /*pos_start*/, uint32_t /*pos_end_exclusive*/) {
  // Intentionally not implemented. This entry point is only exercised by the
  // ALLOCATE prefix-cache slot-copy loopback path, which is out of scope for
  // this adapter. Throwing (instead of a silent stub) ensures any accidental
  // caller crashes loudly at the source rather than deadlocking on a token
  // that will never receive a terminal event.
  throw std::logic_error(
      "[RemoteKVManagerAdapter] migrate() is not implemented "
      "(slot-copy path is unsupported); use start_burst/enqueue/finish_burst "
      "for cross-endpoint KV migration");
}

int RemoteKVManagerAdapter::poll() {
  // Snapshot the pending Kafka ids under the lock; do the getStatus() calls
  // (which take the impl's own lock) OUTSIDE our lock to avoid nested-lock
  // ordering issues, and to keep the critical section small.
  std::vector<uint64_t> toCheck;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    toCheck.reserve(kafkaToGroup_.size());
    for (const auto& [kafkaId, _token] : kafkaToGroup_) {
      toCheck.push_back(kafkaId);
    }
  }

  // Two firings collected under the lock, dispatched OUTSIDE the lock: keeping
  // callbacks out of the critical section preserves the master-worker
  // contract that the callback body may re-enter the adapter (e.g. issue a
  // follow-up migrate()) without deadlocking.
  struct PendingFailure {
    MigrationToken token;
  };
  struct PendingComplete {
    MigrationToken token;
    bool ok;
  };
  std::vector<PendingFailure> failuresToFire;
  std::vector<PendingComplete> completionsToFire;

  for (uint64_t kafkaId : toCheck) {
    const MigrationStatus status = kvManager_->getMigrationStatus(kafkaId);
    if (status == MigrationStatus::IN_PROGRESS ||
        status == MigrationStatus::UNKNOWN) {
      continue;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    auto kit = kafkaToGroup_.find(kafkaId);
    if (kit == kafkaToGroup_.end()) {
      // Already drained (concurrent migrate() unlikely; defensive).
      continue;
    }
    const MigrationToken token = kit->second;
    kafkaToGroup_.erase(kit);

    auto git = groups_.find(token);
    if (git == groups_.end()) {
      // Group already retired (e.g. a race with a duplicate ack). Drop.
      continue;
    }
    MigrationGroup& group = git->second;
    group.pendingKafkaIds.erase(kafkaId);

    if (status == MigrationStatus::FAILED) {
      // Fail-fast: mark sticky, fire onFailed_ once. Keep draining the rest
      // of the group's Kafka ids so kafkaToGroup_ does not leak.
      group.failed = true;
      if (!group.failedReported) {
        group.failedReported = true;
        // TODO(reason): IRemoteKVManager only reports FAILED with no context.
        failuresToFire.push_back({.token = token});
      }
    }

    if (group.pendingKafkaIds.empty() && group.closed) {
      // Terminal for the whole group. Emit ONE completion event per token:
      // success if no per-layer failure ever landed, otherwise nothing extra
      // (onFailed_ already fired above the first time we saw a failure).
      // `closed` gate: for a burst, this prevents firing between enqueue calls
      // where pending happened to transiently empty; for a one-shot migrate(),
      // `closed` is always true so this simplifies to pending.empty().
      if (!group.failed) {
        completionsToFire.push_back({.token = token, .ok = true});
      } else if (!onFailed_) {
        // No onFailed_ registered: fall back to a terminal onComplete_ so the
        // caller still learns the group finished, just with ok=false. Guarded
        // by failedReported == true so we never emit both onFailed_ and this.
        completionsToFire.push_back({.token = token, .ok = false});
      }
      groups_.erase(git);
      drainCv_.notify_all();
    }
  }

  // End-of-poll sweep: pick up any group where finish_burst() flipped `closed`
  // AFTER the pending set had already drained (all Kafka acks landed before
  // finish_burst was called). The per-ack code path above cannot fire in that
  // case because `closed` was still false when the last ack was retired.
  // finish_burst() deliberately does NOT fire the callback itself (see its
  // implementation comment), so this sweep is the sole path that closes the
  // "acks-then-finish" ordering. O(#groups) — bounded by concurrent SUBMITs.
  {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = groups_.begin(); it != groups_.end();) {
      MigrationGroup& g = it->second;
      if (g.closed && g.pendingKafkaIds.empty()) {
        if (!g.failed) {
          completionsToFire.push_back({.token = g.token, .ok = true});
        } else if (!onFailed_) {
          completionsToFire.push_back({.token = g.token, .ok = false});
        }
        it = groups_.erase(it);
        drainCv_.notify_all();
      } else {
        ++it;
      }
    }
  }

  // Fire callbacks outside the lock so re-entrant migrate()/shutdown() calls
  // from user callbacks don't deadlock. Order: failures first (matches the
  // fail-fast intent that failure signal precedes any group closure signal).
  for (const auto& f : failuresToFire) {
    if (onFailed_) {
      onFailed_({.token = f.token, .remote_endpoint_id = -1, .reason = 0});
    }
  }
  for (const auto& c : completionsToFire) {
    if (onComplete_) {
      onComplete_({.token = c.token, .ok = c.ok});
    }
  }

  return static_cast<int>(failuresToFire.size() + completionsToFire.size());
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

  while (!groups_.empty()) {
    if (drainCv_.wait_until(lock, deadline) == std::cv_status::timeout) {
      TT_LOG_WARN(
          "[RemoteKVManagerAdapter] shutdown timeout with {} migration "
          "group(s) still in flight ({} Kafka request(s) outstanding)",
          groups_.size(), kafkaToGroup_.size());
      break;
    }
  }
}

}  // namespace tt::services
