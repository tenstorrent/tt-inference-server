// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/memory_services/paged_memory_manager.hpp"

#include <algorithm>
#include <utility>

#include "config/settings.hpp"
#include "domain/sequence.hpp"
#include "runners/llm_runner/block_manager.hpp"

namespace tt::services {

using Sequence = tt::domain::Sequence;

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

PagedMemoryManager::PagedMemoryManager(
    tt::runners::llm_engine::BlockManager& bm)
    : blockManager(&bm) {}

ManageMemoryStatus PagedMemoryManager::allocateKv(
    const ManageMemoryTask& task, std::vector<int>& outSlotIds) {
  std::vector<int64_t> placeholderTokens(1, 0);  // Hardcoded to use one block
  Sequence seq(task.taskId, blockManager->getBlockSize(),
               std::move(placeholderTokens));

  if (!blockManager->allocate(seq)) {
    return ManageMemoryStatus::WAITING;
  }

  outSlotIds = std::move(seq.getMutableBlockTable());
  return ManageMemoryStatus::SUCCESS;
}

void PagedMemoryManager::deallocateKv(uint32_t taskId,
                                      std::vector<int> slotIds) {
  Sequence seq(taskId, blockManager->getBlockSize(), {});
  seq.getMutableBlockTable() = std::move(slotIds);
  blockManager->deallocate(seq);
}

void PagedMemoryManager::handleRequest(const ManageMemoryTask& task) {
  if (task.action == MemoryManagementAction::MOVE) {
    resultQueue->push(makeResult(task, ManageMemoryStatus::FAILURE));
    return;
  }

  switch (task.action) {
    case MemoryManagementAction::ALLOCATE: {
      // For mock backend, return hardcoded slot ID
      auto llmConfig = tt::config::llmEngineConfig();
      if (llmConfig.runner_type == tt::config::ModelRunnerType::MOCK ||
          llmConfig.runner_type == tt::config::ModelRunnerType::MOCK_PIPELINE) {
        resultQueue->push(makeResult(task, ManageMemoryStatus::SUCCESS, {123}));
        return;
      }

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
      return;
    }
    default:
      resultQueue->push(makeResult(task, ManageMemoryStatus::FAILURE));
  }
}

}  // namespace tt::services
