// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>

namespace tt::services {

/**
 * Service responsible for migrating KV cache data between slots.
 * Coordinates slot copy operations used during prefix-cache routing
 * when a new session needs to start from a cached prefix in another slot.
 */
class MigrationService {
 public:
  MigrationService() = default;
  ~MigrationService() = default;

  MigrationService(const MigrationService&) = delete;
  MigrationService& operator=(const MigrationService&) = delete;

  /**
   * Copy KV cache data from one slot to another.
   *
   * @param sourceSlotId       The slot to copy from.
   * @param destinationSlotId  The slot to copy into.
   * @param numberOfTokens     Number of token positions to copy.
   * @return true on success, false on failure.
   */
  bool copyFromSlot(uint32_t sourceSlotId, uint32_t destinationSlotId,
                    uint32_t numberOfTokens);
};

}  // namespace tt::services
