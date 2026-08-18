// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_migration_executor.hpp"

#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

MooncakeMigrationExecutor::MooncakeMigrationExecutor(
    KvMigrationMultiHostSender& sender)
    : MooncakeMigrationExecutor(
          [&sender](uint64_t uuid, const MigrationRequest& request) {
            return sender.migrate(uuid, request);
          }) {}

MooncakeMigrationExecutor::MooncakeMigrationExecutor(MigrateFn migrate)
    : migrate_(std::move(migrate)) {
  worker_ = std::thread([this] { workerLoop(); });
}

MooncakeMigrationExecutor::~MooncakeMigrationExecutor() {
  std::size_t dropped = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    // Drop queued-but-not-started jobs without firing onDone: the callbacks
    // capture the owning worker, which is tearing down. The scheduler observes
    // these as timeouts (the safe degraded path). The in-flight job, if any,
    // still completes and acks below — its captured state outlives this dtor.
    dropped = queue_.size();
    std::queue<Job> empty;
    queue_.swap(empty);
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  if (dropped > 0) {
    TT_LOG_WARN(
        "[MooncakeMigrationExecutor] dropped {} queued migration(s) on "
        "shutdown",
        dropped);
  }
}

void MooncakeMigrationExecutor::execute(
    uint64_t migrationId, const tt::services::MigrationRequest& request,
    DoneCallback onDone) {
  // Field-for-field copy: services and transport requests share one shape.
  Job job;
  job.id = migrationId;
  job.request = MigrationRequest{
      .src_slot = request.src_slot,
      .dst_slot = request.dst_slot,
      .layer_begin = request.layer_begin,
      .layer_end = request.layer_end,
      .src_position_begin = request.src_position_begin,
      .src_position_end = request.src_position_end,
      .dst_position_begin = request.dst_position_begin,
      .dst_position_end = request.dst_position_end,
  };
  job.onDone = std::move(onDone);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      TT_LOG_WARN(
          "[MooncakeMigrationExecutor] rejecting migration_id={} during "
          "shutdown",
          migrationId);
      return;
    }
    queue_.push(std::move(job));
  }
  cv_.notify_one();
}

void MooncakeMigrationExecutor::workerLoop() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (stopping_) {
        return;
      }
      job = std::move(queue_.front());
      queue_.pop();
    }

    tt::services::MigrationStatus status =
        tt::services::MigrationStatus::FAILED;
    try {
      const bool ok = migrate_ && migrate_(job.id, job.request);
      status = ok ? tt::services::MigrationStatus::SUCCESSFUL
                  : tt::services::MigrationStatus::FAILED;
    } catch (const std::exception& e) {
      TT_LOG_ERROR(
          "[MooncakeMigrationExecutor] migration_id={} threw: {}; reporting "
          "FAILED",
          job.id, e.what());
    } catch (...) {
      TT_LOG_ERROR(
          "[MooncakeMigrationExecutor] migration_id={} threw unknown "
          "exception; reporting FAILED",
          job.id);
    }

    if (job.onDone) {
      job.onDone(status);
    }
  }
}

}  // namespace tt::transport
