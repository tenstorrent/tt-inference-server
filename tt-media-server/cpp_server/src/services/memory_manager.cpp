// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/memory_manager.hpp"

namespace tt::services {

MemoryManager::MemoryManager()
    : requestQueue(ipc::k_memory_request_queue_name,
                   ipc::MEMORY_QUEUE_CAPACITY),
      resultQueue(ipc::k_memory_result_queue_name, ipc::MEMORY_QUEUE_CAPACITY) {
}

MemoryManager::~MemoryManager() = default;

std::optional<domain::ManageMemoryTask> MemoryManager::getRequest() {
  domain::ManageMemoryTask task{};
  if (requestQueue.tryPop(task)) {
    return task;
  }
  return std::nullopt;
}

}  // namespace tt::services
