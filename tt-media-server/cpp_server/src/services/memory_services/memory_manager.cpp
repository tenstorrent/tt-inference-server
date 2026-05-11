// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/memory_services/memory_manager.hpp"

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::services {

MemoryManager::MemoryManager() {
  // Open existing queues created by SessionManager in the main process
  requestQueue = ipc::MemoryRequestQueue::openExisting(
      tt::config::ttMemoryRequestQueueName());
  resultQueue = ipc::MemoryResultQueue::openExisting(
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

void MemoryManager::handleRequest(const domain::ManageMemoryTask& request) {
  switch (request.action) {
    case domain::MemoryManagementAction::ALLOCATE: {
      domain::ManageMemoryResult result{};
      result.taskId = request.taskId;
      result.status = domain::ManageMemoryStatus::SUCCESS;
      result.slotId = 0;
      resultQueue->push(result);
      return;
    }
    case domain::MemoryManagementAction::DEALLOCATE: {
      return;
    }
    default: {
      domain::ManageMemoryResult result{};
      result.taskId = request.taskId;
      result.status = domain::ManageMemoryStatus::FAILURE;
      resultQueue->push(result);
      TT_LOG_WARN("[MemoryManager] Unsupported action {} for taskId={}",
                  static_cast<int>(request.action), request.taskId);
    }
  }
}

}  // namespace tt::services
