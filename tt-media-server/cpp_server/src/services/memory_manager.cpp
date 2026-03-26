// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/memory_manager.hpp"

#include <utility>

#include "runners/llm_runner/block_manager.hpp"

namespace tt::services {

using tt::domain::KvDestination;
using tt::domain::KvMemoryLayout;
using tt::domain::ManageMemoryResult;
using tt::domain::ManageMemoryStatus;
using tt::domain::ManageMemoryTask;
using tt::domain::MemoryManagementAction;

namespace {

ManageMemoryResult makeResult(const ManageMemoryTask& task,
                              ManageMemoryStatus status,
                              std::vector<KvDestination> locations = {}) {
  return ManageMemoryResult{.task_id = task.task_id,
                            .status = status,
                            .memory_locations = std::move(locations)};
}

}  // namespace

MemoryManager::MemoryManager(llm_engine::BlockManager& bm)
    : blockManager(&bm) {}

ManageMemoryStatus MemoryManager::allocateKv(
    std::int32_t inputSeqLen, std::vector<int>& outBlockIds,
    std::vector<KvDestination>& outLocations) {
  if (!blockManager) {
    return ManageMemoryStatus::SUCCESS;
  }

  int blkSize = blockManager->blockSize();
  size_t numBlocks = (static_cast<size_t>(inputSeqLen) +
                      static_cast<size_t>(blkSize) - 1) /
                     static_cast<size_t>(blkSize);

  if (static_cast<size_t>(blockManager->numFreeBlocks()) < numBlocks) {
    return ManageMemoryStatus::WAITING;
  }

  const auto& freeIds = blockManager->freeBlockIds();
  std::vector<int> picked(
      freeIds.begin(),
      freeIds.begin() + static_cast<std::ptrdiff_t>(numBlocks));

  outBlockIds.reserve(numBlocks);
  outLocations.reserve(numBlocks);
  for (int blockId : picked) {
    blockManager->claimBlock(blockId);
    outBlockIds.push_back(blockId);
    // TODO: map blockId to actual device DRAM address
    outLocations.push_back(
        KvDestination{static_cast<uint64_t>(blockId), 0});
  }
  return ManageMemoryStatus::SUCCESS;
}

void MemoryManager::deallocateKv(const std::vector<int>& ids) {
  if (!blockManager) {
    return;
  }
  for (int blockId : ids) {
    blockManager->releaseBlock(blockId);
  }
}

ManageMemoryResult MemoryManager::handle_task(const ManageMemoryTask& task) {
  if (task.action == MemoryManagementAction::MOVE) {
    return makeResult(task, ManageMemoryStatus::FAILURE);
  }

  switch (task.action) {
    case MemoryManagementAction::ALLOCATE: {
      if (reservations.contains(task.task_id.id)) {
        return makeResult(task, ManageMemoryStatus::FAILURE);
      }
      if (task.memory_layout == KvMemoryLayout::PerLayer) {
        return makeResult(task, ManageMemoryStatus::FAILURE);
      }
      if (task.input_seq_len < 0) {
        return makeResult(task, ManageMemoryStatus::FAILURE);
      }

      std::vector<int> blkIds;
      std::vector<KvDestination> locations;
      auto status = allocateKv(task.input_seq_len, blkIds, locations);
      if (status != ManageMemoryStatus::SUCCESS) {
        return makeResult(task, status);
      }

      reservations.insert(task.task_id.id,
                          Reservation{.layout = KvMemoryLayout::Paged,
                                      .blockIds = blkIds,
                                      .locations = locations});
      return makeResult(task, ManageMemoryStatus::SUCCESS,
                        std::move(locations));
    }
    case MemoryManagementAction::DEALLOCATE: {
      auto reservation = reservations.take(task.task_id.id);
      if (!reservation.has_value()) {
        return makeResult(task, ManageMemoryStatus::FAILURE);
      }
      if (reservation->layout != task.memory_layout) {
        reservations.insert(task.task_id.id, std::move(*reservation));
        return makeResult(task, ManageMemoryStatus::FAILURE);
      }

      deallocateKv(reservation->blockIds);
      return makeResult(task, ManageMemoryStatus::SUCCESS);
    }
    default:
      return makeResult(task, ManageMemoryStatus::FAILURE);
  }
}

}  // namespace tt::services
