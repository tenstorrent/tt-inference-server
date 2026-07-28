// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_migration_executor.hpp"

#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

MooncakeMigrationExecutor::MooncakeMigrationExecutor(
    KvMigrationMultiHostSender& sender, std::size_t numThreads)
    : MooncakeMigrationExecutor(
          [&sender](uint64_t uuid, const MigrationRequest& request) {
            return sender.migrate(uuid, request);
          },
          numThreads) {}

MooncakeMigrationExecutor::MooncakeMigrationExecutor(MigrateFn migrate,
                                                     std::size_t numThreads)
    : migrate_(std::move(migrate)) {
  const std::size_t n = (numThreads == 0) ? 1 : numThreads;
  workers_.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    workers_.emplace_back([this] { workerLoop(); });
  }
  if (n > 1) {
    TT_LOG_INFO("[MooncakeMigrationExecutor] started with {} worker threads",
                n);
  }
}

MooncakeMigrationExecutor::~MooncakeMigrationExecutor() {
  std::size_t dropped = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    dropped = queue_.size();
    std::queue<Job> empty;
    queue_.swap(empty);
  }
  cv_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) {
      t.join();
    }
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
