// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/remote_kv_manager_impl.hpp"

#include <utility>

#include "messaging/migration_message.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::services {

RemoteKVManagerImpl::RemoteKVManagerImpl(
    std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer,
    std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer,
    std::size_t migrationWorkerPoolSize, std::chrono::milliseconds timeout,
    std::chrono::milliseconds sweepInterval, int drainPollMs)
    : requestProducer(std::move(requestProducer)),
      ackConsumer(std::move(ackConsumer)),
      migrationWorkerPoolSize(migrationWorkerPoolSize),
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
  if (this->migrationWorkerPoolSize == 0) {
    TT_LOG_WARN(
        "[RemoteKVManagerImpl] migrationWorkerPoolSize=0; clamping to 1. "
        "Downloads would never reach COMPLETED with pool size 0.");
    this->migrationWorkerPoolSize = 1;
  }
  running.store(true, std::memory_order_relaxed);
  lastSweep = std::chrono::steady_clock::now();
  drainThread = std::thread([this] { drainLoop(); });
  TT_LOG_INFO(
      "[RemoteKVManagerImpl] started (workerPool={}, timeout={}ms, sweep={}ms, "
      "drainPoll={}ms)",
      this->migrationWorkerPoolSize, this->timeout.count(),
      this->sweepInterval.count(), this->drainPollMs);
}

RemoteKVManagerImpl::~RemoteKVManagerImpl() {
  running.store(false, std::memory_order_relaxed);
  if (drainThread.joinable()) {
    drainThread.join();
  }
  TT_LOG_INFO("[RemoteKVManagerImpl] stopped");
}

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
      .layer_id = request.layer_id,
      .position_start = request.position_start,
      .position_end = request.position_end,
  };
  const std::string payload = tt::messaging::serialize(msg);

  bool sent = false;
  std::string err;
  if (requestProducer) {
    sent = requestProducer->send(payload, &err);
  } else {
    err = "no producer";
  }

  if (!sent) {
    // Roll the migration straight to FAILED so callers don't wait `timeout`
    // for a request that never made it onto the wire.
    std::lock_guard<std::mutex> lock(mtx);
    auto it = migrations.find(id);
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

MigrationStatus RemoteKVManagerImpl::getStatus(uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = migrations.find(migrationId);
  if (it == migrations.end()) {
    return MigrationStatus::UNKNOWN;
  }
  return it->second.status;
}

uint64_t RemoteKVManagerImpl::downloadFromStore(
    const DownloadKVRequest& request) {
  const uint64_t id = tt::utils::MigrationIDGenerator::generate();
  const auto now = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(mtx);
    auto [it, inserted] = downloads.emplace(
        id, DownloadState{KVTransferStatus::IN_PROGRESS,
                          /*usablePrefixCount=*/0, now});
    if (!inserted) {
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] id collision on download transfer_id={}; "
          "returning existing record (status={})",
          id, static_cast<int>(it->second.status));
      return id;
    }
  }

  TT_LOG_INFO(
      "[RemoteKVManagerImpl] downloadFromStore (no-op) transfer_id={} "
      "dst_slot={} blocks={} pool={}",
      id, request.dstSlot, request.blocks.size(), migrationWorkerPoolSize);
  return id;
}

KVTransferResult RemoteKVManagerImpl::getDownloadResult(
    uint64_t transferId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = downloads.find(transferId);
  if (it == downloads.end()) {
    return KVTransferResult{KVTransferStatus::UNKNOWN, 0};
  }
  return KVTransferResult{it->second.status, it->second.usablePrefixCount};
}

void RemoteKVManagerImpl::offloadToStore(const OffloadKVRequest& request) {
  TT_LOG_INFO(
      "[RemoteKVManagerImpl] offloadToStore (no-op, fire-and-forget) "
      "src_slot={} blocks={}",
      request.srcSlot, request.blocks.size());
}

void RemoteKVManagerImpl::drainLoop() {
  TT_LOG_INFO("[RemoteKVManagerImpl] drain loop entered");

  while (running.load(std::memory_order_relaxed)) {
    if (ackConsumer) {
      auto msg = ackConsumer->receive(drainPollMs);
      if (msg.has_value()) {
        auto parsed = tt::messaging::parseMigrationResponse(*msg);
        if (!parsed.has_value()) {
          TT_LOG_WARN(
              "[RemoteKVManagerImpl] dropping unparseable ack payload: {}",
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

    const auto now = std::chrono::steady_clock::now();
    if (now - lastSweep >= sweepInterval) {
      std::lock_guard<std::mutex> lock(mtx);
      sweepLocked(now);
      lastSweep = now;
    }
  }

  TT_LOG_INFO("[RemoteKVManagerImpl] drain loop exited");
}

void RemoteKVManagerImpl::sweepLocked(
    std::chrono::steady_clock::time_point now) {
  size_t timedOutMigrations = 0;
  for (auto& [id, state] : migrations) {
    if (state.status == MigrationStatus::IN_PROGRESS &&
        now - state.submittedAt >= timeout) {
      state.status = MigrationStatus::FAILED;
      ++timedOutMigrations;
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] migration_id={} timed out after {}ms; marked "
          "FAILED",
          id, timeout.count());
    }
  }
  if (timedOutMigrations > 0) {
    TT_LOG_INFO("[RemoteKVManagerImpl] sweeper timed out {} migration(s)",
                timedOutMigrations);
  }

  size_t timedOutDownloads = 0;
  for (auto& [id, state] : downloads) {
    if (state.status == KVTransferStatus::IN_PROGRESS &&
        now - state.submittedAt >= timeout) {
      state.status = KVTransferStatus::FAILED;
      state.usablePrefixCount = 0;
      ++timedOutDownloads;
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] download transfer_id={} timed out after {}ms; "
          "marked FAILED",
          id, timeout.count());
    }
  }
  if (timedOutDownloads > 0) {
    TT_LOG_INFO("[RemoteKVManagerImpl] sweeper timed out {} download(s)",
                timedOutDownloads);
  }
}

}  // namespace tt::services
