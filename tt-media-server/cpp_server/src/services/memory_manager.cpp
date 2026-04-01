// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/memory_manager.hpp"

#include "utils/logger.hpp"

namespace tt::services {

MemoryManager::MemoryManager() {
  // Open existing queues created by SessionManager in the main process
  requestQueue =
      ipc::MemoryRequestQueue::openExisting(ipc::k_memory_request_queue_name);
  resultQueue =
      ipc::MemoryResultQueue::openExisting(ipc::k_memory_result_queue_name);

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

}  // namespace tt::services
