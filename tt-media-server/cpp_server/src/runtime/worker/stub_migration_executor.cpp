// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/worker/stub_migration_executor.hpp"

#include "utils/logger.hpp"

namespace tt::worker {

StubMigrationExecutor::StubMigrationExecutor(
    tt::services::MigrationStatus result)
    : result(result) {}

void StubMigrationExecutor::execute(uint64_t migrationId,
                                    const tt::services::MigrationRequest& req,
                                    DoneCallback onDone) {
  TT_LOG_INFO(
      "[DryRunMigrationExecutor] migration_id={} src_slot={} dst_slot={} "
      "layers=[{}..{}) src_pos=[{}..{}) dst_pos=[{}..{}) -> {}",
      migrationId, req.src_slot, req.dst_slot, req.layer_begin, req.layer_end,
      req.src_position_begin, req.src_position_end, req.dst_position_begin,
      req.dst_position_end, static_cast<int>(result));

  if (onDone) {
    onDone(result);
  }
}

}  // namespace tt::worker
