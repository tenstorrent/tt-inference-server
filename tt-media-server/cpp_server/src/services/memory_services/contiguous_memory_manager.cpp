// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/memory_services/contiguous_memory_manager.hpp"

#include "domain/manage_memory.hpp"

namespace tt::services {

using tt::domain::ManageMemoryResult;
using tt::domain::ManageMemoryStatus;
using tt::domain::ManageMemoryTask;

void ContiguousMemoryManager::handleRequest(const ManageMemoryTask& task) {
  ManageMemoryResult result{};
  result.taskId = task.taskId;
  // TODO return a proper slot id here
  result.slotIds = {static_cast<std::uint32_t>(123)};
  if (task.action != domain::MemoryManagementAction::DEALLOCATE) {
    result.status = domain::ManageMemoryStatus::SUCCESS;
  }
  resultQueue->push(result);
}

}  // namespace tt::services
