// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>

#include "services/remote_kv_manager.hpp"

namespace tt::worker {

/**
 * Seam that hides the byte-moving implementation from KvMigrationWorker.
 *
 * The worker hands every parsed MigrationRequest to an executor and is
 * otherwise indifferent to how the bytes actually move. Production wires
 * the Mooncake-backed implementation; tests wire a mock; early bring-up
 * uses StubMigrationExecutor (always succeeds).
 *
 * Threading contract:
 *   - execute() returns immediately. It MUST NOT block waiting for the
 *     migration to finish.
 *   - onDone is invoked exactly once with the terminal status. It MAY
 *     fire synchronously from within execute() (e.g. the Stub) or from
 *     any other thread the executor chooses.
 *   - The executor MUST NOT invoke onDone after the worker that owns it
 *     has been destroyed. Synchronous executors satisfy this trivially.
 *     Asynchronous executors are responsible for draining their pending
 *     callbacks before destruction.
 */
class IMigrationExecutor {
 public:
  using DoneCallback =
      std::function<void(tt::services::MigrationStatus status)>;

  virtual ~IMigrationExecutor() = default;

  /**
   * Submit a migration for execution.
   *
   * @param migrationId  Identifier minted by RemoteKVManagerImpl. Echoed
   *   back to the caller via onDone and used for logging only; the
   *   executor itself does not need to disambiguate by id.
   * @param request      KV-cache slice to move.
   * @param onDone       Invoked exactly once with the terminal status.
   */
  virtual void execute(uint64_t migrationId,
                       const tt::services::MigrationRequest& request,
                       DoneCallback onDone) = 0;
};

}  // namespace tt::worker
