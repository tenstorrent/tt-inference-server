// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>

#include "runtime/worker/migration_executor.hpp"
#include "services/remote_kv_manager.hpp"

namespace tt::worker {

/**
 * Placeholder executor used until the real Mooncake-backed executor lands.
 *
 * Behaviour: logs the request and invokes onDone synchronously with the
 * configured terminal status (default SUCCESSFUL). The "synchronous fast
 * path" is intentional - the Stub does no real work, so spawning a
 * thread per migration would add latency without benefit. Real executors
 * are free to invoke onDone asynchronously.
 *
 * Thread-safety: stateless apart from a single immutable status, so
 * concurrent execute() calls from multiple worker threads are safe.
 */
class StubMigrationExecutor : public IMigrationExecutor {
 public:
  explicit StubMigrationExecutor(tt::services::MigrationStatus result =
                                     tt::services::MigrationStatus::SUCCESSFUL);

  void execute(uint64_t migrationId,
               const tt::services::MigrationRequest& request,
               DoneCallback onDone) override;

 private:
  tt::services::MigrationStatus result;
};

}  // namespace tt::worker
