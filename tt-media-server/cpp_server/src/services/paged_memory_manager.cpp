// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/paged_memory_manager.hpp"

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

PagedMemoryManager::PagedMemoryManager(llm_engine::BlockManager& bm)
    : blockManager(&bm) {}

ManageMemoryStatus PagedMemoryManager::allocateKv(
    const ManageMemoryTask& task, std::vector<int>& outSlotIds) {
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

void PagedMemoryManager::deallocateKv(const domain::TaskID& taskId,
                                      std::vector<int> slotIds) {
  llm_engine::Sequence seq(taskId, blockManager->blockSize(), {});
  seq.blockTable = std::move(slotIds);
  blockManager->deallocate(seq);
}

void PagedMemoryManager::handleRequest(const ManageMemoryTask& task) {
  if (task.action == MemoryManagementAction::MOVE) {
    resultQueue->push(makeResult(task, ManageMemoryStatus::FAILURE));
    return;
  }

  switch (task.action) {
    case MemoryManagementAction::ALLOCATE: {
      std::vector<int> slotIds;
      auto status = allocateKv(task, slotIds);
      if (status == ManageMemoryStatus::WAITING) {
        requestQueue->push(task, /*priority=*/1);
        return;
      }
      if (status != ManageMemoryStatus::SUCCESS) {
        resultQueue->push(makeResult(task, status));
        return;
      }

      std::vector<std::uint32_t> resultIds(slotIds.begin(), slotIds.end());
      resultQueue->push(
          makeResult(task, ManageMemoryStatus::SUCCESS, std::move(resultIds)));
      return;
    }
    case MemoryManagementAction::DEALLOCATE: {
      std::vector<int> slotIds(task.slotIds.begin(), task.slotIds.end());
      deallocateKv(task.taskId, std::move(slotIds));
      resultQueue->push(makeResult(task, ManageMemoryStatus::SUCCESS));
      return;
    }
    default:
      resultQueue->push(makeResult(task, ManageMemoryStatus::FAILURE));
  }
}

}  // namespace tt::services
