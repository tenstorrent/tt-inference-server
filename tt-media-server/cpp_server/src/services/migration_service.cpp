// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/migration_service.hpp"

#include <chrono>
#include <thread>

#include "utils/logger.hpp"

namespace tt::services {

bool MigrationService::copyFromSlot(uint32_t sourceSlotId,
                                    uint32_t destinationSlotId,
                                    uint32_t numberOfTokens) {
  TT_LOG_INFO(
      "[MigrationService] copyFromSlot: src={} dst={} tokens={} — "
      "sleeping 500ms (stub)",
      sourceSlotId, destinationSlotId, numberOfTokens);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  TT_LOG_INFO("[MigrationService] copyFromSlot: done src={} dst={}",
              sourceSlotId, destinationSlotId);
  return true;
}

}  // namespace tt::services
