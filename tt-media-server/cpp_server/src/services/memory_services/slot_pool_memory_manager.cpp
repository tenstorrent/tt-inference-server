// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/memory_services/slot_pool_memory_manager.hpp"

#include <string>

#include "utils/event_recorder.hpp"
#include "utils/logger.hpp"

namespace tt::services {

using tt::domain::ManageMemoryResult;
using tt::domain::ManageMemoryStatus;
using tt::domain::ManageMemoryTask;
using tt::domain::MemoryManagementAction;

SlotPoolMemoryManager::SlotPoolMemoryManager(uint32_t poolSize)
    : slotPoolSize(poolSize) {
  for (uint32_t i = 0; i < poolSize; ++i) {
    freeSlots.insert(i);
  }
  TT_LOG_INFO("[SlotPoolMemoryManager] Initialized with {} slots", poolSize);

  auto& recorder = tt::utils::EventRecorder::instance();
  recorder.record("MM", "POOL_INITIALIZED",
                  "\"pool_size\":" + std::to_string(poolSize));
}

void SlotPoolMemoryManager::handleRequest(const ManageMemoryTask& task) {
  auto& recorder = tt::utils::EventRecorder::instance();

  switch (task.action) {
    case MemoryManagementAction::ALLOCATE: {
      if (freeSlots.empty()) {
        TT_LOG_DEBUG(
            "[SlotPoolMemoryManager] ALLOCATE taskId={}: pool exhausted "
            "(allocated={})",
            task.taskId, allocatedSlots.size());

        recorder.record(
            "MM", "ALLOC_EXHAUSTED",
            "\"task_id\":" + std::to_string(task.taskId) +
                ",\"allocated\":" + std::to_string(allocatedSlots.size()));

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
          "[SlotPoolMemoryManager] ALLOCATE taskId={}: assigned slot={}, "
          "free={}, allocated={}",
          task.taskId, slotId, freeSlots.size(), allocatedSlots.size());

      recorder.record(
          "MM", "SLOT_ALLOCATED",
          "\"task_id\":" + std::to_string(task.taskId) +
              ",\"slot_id\":" + std::to_string(slotId) +
              ",\"free\":" + std::to_string(freeSlots.size()) +
              ",\"allocated\":" + std::to_string(allocatedSlots.size()));

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
              "[SlotPoolMemoryManager] DEALLOCATE taskId={}: freed slot={}, "
              "free={}, allocated={}",
              task.taskId, slotId, freeSlots.size(), allocatedSlots.size());

          recorder.record(
              "MM", "SLOT_DEALLOCATED",
              "\"task_id\":" + std::to_string(task.taskId) +
                  ",\"slot_id\":" + std::to_string(slotId) +
                  ",\"free\":" + std::to_string(freeSlots.size()) +
                  ",\"allocated\":" + std::to_string(allocatedSlots.size()));
        } else {
          TT_LOG_WARN(
              "[SlotPoolMemoryManager] DEALLOCATE taskId={}: slot={} was not "
              "allocated",
              task.taskId, slotId);

          recorder.record(
              "MM", "DEALLOC_UNKNOWN_SLOT",
              "\"task_id\":" + std::to_string(task.taskId) +
                  ",\"slot_id\":" + std::to_string(slotId));
        }
      }
      return;
    }

    default: {
      TT_LOG_WARN("[SlotPoolMemoryManager] Unsupported action {} for taskId={}",
                  static_cast<int>(task.action), task.taskId);
      ManageMemoryResult result{};
      result.taskId = task.taskId;
      result.status = ManageMemoryStatus::FAILURE;
      resultQueue->push(result);
    }
  }
}

}  // namespace tt::services
