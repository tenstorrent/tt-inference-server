// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <set>

#include "services/memory_services/memory_manager.hpp"

namespace tt::services {

class ContiguousMemoryManager : public MemoryManager {
 public:
  explicit ContiguousMemoryManager(uint32_t poolSize);

  void handleRequest(const domain::ManageMemoryTask& request) override;

  uint32_t getPoolSize() const { return slotPoolSize; }
  uint32_t getFreeCount() const {
    return static_cast<uint32_t>(freeSlots.size());
  }

 private:
  uint32_t slotPoolSize;
  std::set<uint32_t> freeSlots;
  std::set<uint32_t> allocatedSlots;
};

}  // namespace tt::services
