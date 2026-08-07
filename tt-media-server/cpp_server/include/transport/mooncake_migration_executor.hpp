// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "runtime/worker/migration_executor.hpp"
#include "services/remote_kv_manager.hpp"
#include "transport/kv_migration_multi_host_sender.hpp"
#include "transport/kv_table_adapter.hpp"

namespace tt::transport {

/**
 * @brief The real, Mooncake-backed IMigrationExecutor — the interface where the
 *        Kafka trigger meets the KV data plane.
 *
 * KvMigrationWorker hands every parsed request to an IMigrationExecutor and is
 * indifferent to how the bytes move. This implementation drives the migration
 * across the decode cluster via KvMigrationMultiHostSender (which fans out to
 * every decode host the request touches). It replaces StubMigrationExecutor.
 *
 * No request-shape translation: tt::services::MigrationRequest mirrors
 * tt::transport::MigrationRequest field-for-field, so the only mapping is a
 * by-name struct copy.
 *
 * Threading: KvMigrationMultiHostSender::migrate() is blocking (it reads
 * device DRAM, issues one-sided writes, and waits on per-host acks), but
 * IMigrationExecutor::execute() must return immediately. So execute() enqueues
 * the request and a single background thread runs migrations one at a time
 * (the per-host senders/channels are not designed for concurrent migrate()
 * calls). On terminal completion the background thread invokes onDone with
 * SUCCESSFUL (migrate() == true) or FAILED (false, threw, or shutting down).
 * The destructor stops the thread and drains in-flight work, so onDone never
 * fires after this object is gone.
 */
class MooncakeMigrationExecutor : public tt::worker::IMigrationExecutor {
 public:
  /// The callable that performs one migration: returns true iff every decode
  /// host the request touches completed. Injected so the executor's threading
  /// and status mapping can be tested without the full transport stack;
  /// production binds it to KvMigrationMultiHostSender::migrate.
  using MigrateFn =
      std::function<bool(uint64_t uuid, const MigrationRequest& request)>;

  /// Production: drive the real multi-host data plane. `sender` must outlive
  /// this executor.
  explicit MooncakeMigrationExecutor(KvMigrationMultiHostSender& sender);

  /// Test/DI: drive an arbitrary migrate callable.
  explicit MooncakeMigrationExecutor(MigrateFn migrate);

  ~MooncakeMigrationExecutor() override;

  MooncakeMigrationExecutor(const MooncakeMigrationExecutor&) = delete;
  MooncakeMigrationExecutor& operator=(const MooncakeMigrationExecutor&) =
      delete;

  void execute(uint64_t migrationId,
               const tt::services::MigrationRequest& request,
               DoneCallback onDone) override;

 private:
  struct Job {
    uint64_t id = 0;
    MigrationRequest request;
    DoneCallback onDone;
  };

  void workerLoop();

  MigrateFn migrate_;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<Job> queue_;
  bool stopping_ = false;
  std::thread worker_;
};

}  // namespace tt::transport
