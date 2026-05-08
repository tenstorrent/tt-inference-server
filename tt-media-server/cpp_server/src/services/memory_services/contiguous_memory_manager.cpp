// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/memory_services/contiguous_memory_manager.hpp"

#include "domain/manage_memory.hpp"
#include "utils/logger.hpp"

namespace tt::services {

using tt::domain::ManageMemoryResult;
using tt::domain::ManageMemoryStatus;
using tt::domain::ManageMemoryTask;
using tt::domain::MemoryManagementAction;

ContiguousMemoryManager::ContiguousMemoryManager(uint32_t poolSize)
    : slotPoolSize(poolSize) {
  for (uint32_t i = 0; i < poolSize; ++i) {
    freeSlots.insert(i);
  }
  TT_LOG_INFO("[ContiguousMemoryManager] Initialized with {} slots", poolSize);
}

void ContiguousMemoryManager::handleRequest(const ManageMemoryTask& task) {
  switch (task.action) {
    case MemoryManagementAction::ALLOCATE: {
      if (freeSlots.empty()) {
        TT_LOG_DEBUG(
            "[ContiguousMemoryManager] ALLOCATE taskId={}: pool exhausted "
            "(allocated={})",
            task.taskId, allocatedSlots.size());

        ManageMemoryResult result{};
        result.taskId = task.taskId;
        result.status = ManageMemoryStatus::FAILURE;
        resultQueue->push(result);
        return;
      }

      uint32_t slotId = *freeSlots.begin();
      freeSlots.erase(freeSlots.begin());
      allocatedSlots.insert(slotId);

      TT_LOG_DEBUG(
          "[ContiguousMemoryManager] ALLOCATE taskId={}: assigned slot={}, "
          "free={}, allocated={}",
          task.taskId, slotId, freeSlots.size(), allocatedSlots.size());

      ManageMemoryResult result{};
      result.taskId = task.taskId;
      result.status = ManageMemoryStatus::SUCCESS;
      result.slotIds = {slotId};
      resultQueue->push(result);
      return;
    }

    case MemoryManagementAction::DEALLOCATE: {
      for (uint32_t slotId : task.slotIds) {
        if (allocatedSlots.erase(slotId) > 0) {
          freeSlots.insert(slotId);
          TT_LOG_DEBUG(
              "[ContiguousMemoryManager] DEALLOCATE taskId={}: freed slot={}, "
              "free={}, allocated={}",
              task.taskId, slotId, freeSlots.size(), allocatedSlots.size());
        } else {
          TT_LOG_WARN(
              "[ContiguousMemoryManager] DEALLOCATE taskId={}: slot={} was "
              "not allocated",
              task.taskId, slotId);
        }
      }
      return;
    }

    default: {
      TT_LOG_WARN(
          "[ContiguousMemoryManager] Unsupported action {} for taskId={}",
          static_cast<int>(task.action), task.taskId);
      ManageMemoryResult result{};
      result.taskId = task.taskId;
      result.status = ManageMemoryStatus::FAILURE;
      resultQueue->push(result);
    }
  }
}

}  // namespace tt::services
