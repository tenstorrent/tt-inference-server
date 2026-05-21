// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/memory_services/memory_manager.hpp"

#include "config/settings.hpp"
#include "domain/slot_types.hpp"
#include "utils/logger.hpp"

namespace tt::services {

MemoryManager::MemoryManager() {
  // Open existing queues created by SessionManager in the main process
  requestQueue = ipc::boost::MemoryRequestQueue::openExisting(
      tt::config::ttMemoryRequestQueueName());
  resultQueue = ipc::boost::MemoryResultQueue::openExisting(
      tt::config::ttMemoryResultQueueName());

  if (!requestQueue || !resultQueue) {
    TT_LOG_ERROR(
        "[MemoryManager] Failed to open memory queues. SessionManager should "
        "have created them.");
    throw std::runtime_error("Memory queues not available");
  }

  TT_LOG_INFO("[MemoryManager] Opened memory management IPC queues");
}

MemoryManager::~MemoryManager() {
  TT_LOG_INFO("[MemoryManager] Shutting down");
}

std::optional<domain::ManageMemoryTask> MemoryManager::getRequest() {
  domain::ManageMemoryTask task{};
  if (requestQueue->tryPop(task)) {
    return task;
  }
  return std::nullopt;
}

void MemoryManager::replyAllocateSuccess(uint32_t taskId, uint32_t slotId) {
  domain::ManageMemoryResult result{};
  result.taskId = taskId;
  result.status = domain::ManageMemoryStatus::SUCCESS;
  result.slotId = slotId;
  resultQueue->push(result);
}

void MemoryManager::replyAllocateFailure(uint32_t taskId) {
  domain::ManageMemoryResult result{};
  result.taskId = taskId;
  result.status = domain::ManageMemoryStatus::FAILURE;
  result.slotId = tt::domain::INVALID_SLOT_ID;
  resultQueue->push(result);
}

}  // namespace tt::services
