// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/memory_manager.hpp"

#include <utility>

#include "runners/llm_runner/block_manager.hpp"
#include "runners/llm_runner/sequence.hpp"

namespace tt::services {

using tt::domain::ManageMemoryResult;
using tt::domain::ManageMemoryStatus;
using tt::domain::ManageMemoryTask;
using tt::domain::MemoryManagementAction;

namespace {

ManageMemoryResult makeResult(const ManageMemoryTask& task,
                              ManageMemoryStatus status,
                              std::vector<std::uint32_t> slotIds = {}) {
  return ManageMemoryResult{
      .taskId = task.taskId, .status = status, .slotIds = std::move(slotIds)};
}

}  // namespace

MemoryManager::MemoryManager(llm_engine::BlockManager& bm)
    : blockManager(&bm) {}

ManageMemoryStatus MemoryManager::allocateKv(const ManageMemoryTask& task,
                                             std::vector<int>& outSlotIds) {
  if (!blockManager) {
    return ManageMemoryStatus::SUCCESS;
  }

  std::vector<int64_t> placeholderTokens(static_cast<size_t>(task.inputSeqLen),
                                         0);
  llm_engine::Sequence seq(task.taskId, blockManager->blockSize(),
                           std::move(placeholderTokens));

  if (!blockManager->allocate(seq)) {
    return ManageMemoryStatus::WAITING;
  }

  outSlotIds = std::move(seq.blockTable);
  return ManageMemoryStatus::SUCCESS;
}

void MemoryManager::deallocateKv(const domain::TaskID& taskId,
                                 std::vector<int> slotIds) {
  if (!blockManager) {
    return;
  }
  llm_engine::Sequence seq(taskId, blockManager->blockSize(), {});
  seq.blockTable = std::move(slotIds);
  blockManager->deallocate(seq);
}

ManageMemoryResult MemoryManager::handleTask(const ManageMemoryTask& task) {
  if (task.action == MemoryManagementAction::MOVE) {
    return makeResult(task, ManageMemoryStatus::FAILURE);
  }

  switch (task.action) {
    case MemoryManagementAction::ALLOCATE: {
      std::vector<int> slotIds;
      auto status = allocateKv(task, slotIds);
      if (status != ManageMemoryStatus::SUCCESS) {
        return makeResult(task, status);
      }

      std::vector<std::uint32_t> resultIds(slotIds.begin(), slotIds.end());
      return makeResult(task, ManageMemoryStatus::SUCCESS,
                        std::move(resultIds));
    }
    case MemoryManagementAction::DEALLOCATE: {
      std::vector<int> slotIds(task.slotIds.begin(), task.slotIds.end());
      deallocateKv(task.taskId, std::move(slotIds));
      return makeResult(task, ManageMemoryStatus::SUCCESS);
    }
    default:
      return makeResult(task, ManageMemoryStatus::FAILURE);
  }
}

}  // namespace tt::services
