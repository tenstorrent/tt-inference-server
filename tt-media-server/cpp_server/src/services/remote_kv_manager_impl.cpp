// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/remote_kv_manager_impl.hpp"

#include <utility>

#include "messaging/migration_message.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::services {

/**
 * Constructor for RemoteKVManagerImpl. Initializes the manager with the given
 * parameters.
 *
 * @param requestProducer Kafka producer wired to the migration-request topic.
 * @param ackConsumer Kafka consumer subscribed to the migration-ack topic, with
 * a unique group.id.
 * @param timeout Max age of an IN_PROGRESS migration before the sweeper marks
 * it FAILED. Default 60s.
 * @param sweepInterval How often the drain thread runs the timeout sweep.
 * Default 5s.
 * @param drainPollMs Per-iteration poll timeout passed to the consumer. Default
 * 100ms.
 */
RemoteKVManagerImpl::RemoteKVManagerImpl(
    std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer,
    std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer,
    std::chrono::milliseconds timeout, std::chrono::milliseconds sweepInterval,
    int drainPollMs, LayerToPartition layerToPartition)
    : requestProducer(std::move(requestProducer)),
      ackConsumer(std::move(ackConsumer)),
      layerToPartition(std::move(layerToPartition)),
      timeout(timeout),
      sweepInterval(sweepInterval),
      drainPollMs(drainPollMs) {
  if (!this->requestProducer) {
    TT_LOG_ERROR(
        "[RemoteKVManagerImpl] null requestProducer; migrate() will fail every "
        "call");
  }
  if (!this->ackConsumer) {
    TT_LOG_ERROR(
        "[RemoteKVManagerImpl] null ackConsumer; statuses will never "
        "transition out of IN_PROGRESS via ack");
  }
  running.store(true, std::memory_order_relaxed);
  lastSweep = std::chrono::steady_clock::now();
  drainThread = std::thread([this] { drainLoop(); });
  TT_LOG_INFO(
      "[RemoteKVManagerImpl] started (timeout={}ms, sweep={}ms, "
      "drainPoll={}ms)",
      this->timeout.count(), this->sweepInterval.count(), this->drainPollMs);
}

/**
 * Destructor for RemoteKVManagerImpl. Stops the drain loop and joins the drain
 * thread.
 */
RemoteKVManagerImpl::~RemoteKVManagerImpl() {
  running.store(false, std::memory_order_relaxed);
  if (drainThread.joinable()) {
    drainThread.join();
  }
  TT_LOG_INFO("[RemoteKVManagerImpl] stopped");
}

/**
 * Method migrate is used to migrate a key-value pair from one slot to another.
 * It generates a new migration id and adds a new migration state to the
 * migrations map with status IN_PROGRESS. It also sends a migration request
 * message to the request topic.
 *
 * Returns the migration id.
 */
uint64_t RemoteKVManagerImpl::migrate(const MigrationRequest& request) {
  const uint64_t id = tt::utils::MigrationIDGenerator::generate();
  const auto now = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(mtx);
    // 64-bit random id collisions are astronomically unlikely; if one ever
    // occurs we keep the older record (insert is a no-op) and the caller
    // will observe whatever state that older migration is in. This is safer
    // than overwriting an in-flight record.
    auto [it, inserted] = migrations.emplace(
        id, MigrationState{MigrationStatus::IN_PROGRESS, now});
    if (!inserted) {
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] id collision on migration_id={}; returning "
          "existing record (status={})",
          id, static_cast<int>(it->second.status));
      return id;
    }
  }

  const tt::messaging::MigrationRequestMessage msg{
      .migration_id = id,
      .src_slot = request.src_slot,
      .dst_slot = request.dst_slot,
      .layer_begin = request.layer_begin,
      .layer_end = request.layer_end,
      .src_position_begin = request.src_position_begin,
      .src_position_end = request.src_position_end,
      .dst_position_begin = request.dst_position_begin,
      .dst_position_end = request.dst_position_end,
  };
  const std::string payload = tt::messaging::serialize(msg);

  bool sent = false;
  std::string err;
  if (requestProducer) {
    if (layerToPartition) {
      const int32_t partition = layerToPartition(request.layer_begin);
      sent = partition >= 0 ? requestProducer->send(payload, partition, &err)
                            : requestProducer->send(payload, &err);
    } else {
      sent = requestProducer->send(payload, &err);
    }
  } else {
    err = "no producer";
  }

  if (!sent) {
    // Roll the migration straight to FAILED so callers don't wait `timeout`
    // for a request that never made it onto the wire.
    std::lock_guard<std::mutex> lock(mtx);
    auto it = migrations.find(id);

    // We need check the migration status to avoid overwriting a successful or
    // failed migration.
    if (it != migrations.end() &&
        it->second.status == MigrationStatus::IN_PROGRESS) {
      it->second.status = MigrationStatus::FAILED;
    }
    TT_LOG_ERROR(
        "[RemoteKVManagerImpl] producer.send failed for migration_id={}: {}",
        id, err);
  }

  return id;
}

/**
 * Method getMigrationStatus is used to get the status of a migration for given
 * migrationId.
 */
MigrationStatus RemoteKVManagerImpl::getMigrationStatus(
    uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = migrations.find(migrationId);
  if (it == migrations.end()) {
    return MigrationStatus::UNKNOWN;
  }

  return it->second.status;
}

/**
 * Method drainLoop is used to drain the acknowledgments from the ack topic and
 * update the migration status. It also sweeps the migrations and marks any
 * migration whose request was issued more than `timeout` ago and is still
 * IN_PROGRESS as FAILED.
 *
 * Runs as a background thread.
 */
void RemoteKVManagerImpl::drainLoop() {
  TT_LOG_INFO("[RemoteKVManagerImpl] drain loop entered");

  while (running.load(std::memory_order_relaxed)) {
    // Drain the acknowledgments from the ack topic and update the migration
    // status.
    if (ackConsumer) {
      auto msg = ackConsumer->receive(drainPollMs);
      if (msg.has_value()) {
        auto parsed = tt::messaging::parseMigrationResponse(*msg);
        if (!parsed.has_value()) {
          TT_LOG_WARN(
              "[RemoteKVManagerImpl] dropping unparsable ack payload: {}",
              *msg);
        } else {
          std::lock_guard<std::mutex> lock(mtx);
          auto it = migrations.find(parsed->migration_id);
          if (it == migrations.end()) {
            TT_LOG_WARN(
                "[RemoteKVManagerImpl] ack for unknown migration_id={}; "
                "ignoring",
                parsed->migration_id);
          } else if (it->second.status != MigrationStatus::IN_PROGRESS) {
            TT_LOG_DEBUG(
                "[RemoteKVManagerImpl] ack for already-terminal migration_id="
                "{}, status={}; ignoring",
                parsed->migration_id, static_cast<int>(it->second.status));
          } else {
            it->second.status = parsed->status;
          }
        }
      }
    } else {
      // No consumer: still respect the poll cadence so the loop doesn't spin.
      std::this_thread::sleep_for(std::chrono::milliseconds(drainPollMs));
    }

    // Sweep the migrations and mark any migration whose request was issued more
    // than `timeout`
    const auto now = std::chrono::steady_clock::now();
    if (now - lastSweep >= sweepInterval) {
      std::lock_guard<std::mutex> lock(mtx);
      sweepLocked(now);
      lastSweep = now;
    }
  }

  TT_LOG_INFO("[RemoteKVManagerImpl] drain loop exited");
}

/**
 * Method sweepLocked is used to sweep the migrations and mark any migration
 * whose request was issued more than `timeout` ago and is still IN_PROGRESS as
 * FAILED.
 */
void RemoteKVManagerImpl::sweepLocked(
    std::chrono::steady_clock::time_point now) {
  size_t timedOut = 0;
  for (auto& [id, migrationState] : migrations) {
    if (migrationState.status == MigrationStatus::IN_PROGRESS &&
        now - migrationState.submittedAt >= timeout) {
      migrationState.status = MigrationStatus::FAILED;
      ++timedOut;
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] migration_id={} timed out after {}ms; marked "
          "FAILED",
          id, timeout.count());
    }
  }
  if (timedOut > 0) {
    TT_LOG_INFO("[RemoteKVManagerImpl] sweeper timed out {} migration(s)",
                timedOut);
  }
}

}  // namespace tt::services
