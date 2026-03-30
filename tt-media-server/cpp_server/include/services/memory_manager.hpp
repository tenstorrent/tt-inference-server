// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <vector>

#include "domain/manage_memory.hpp"

namespace llm_engine {
class BlockManager;
}

namespace tt::services {

class MemoryManager {
 public:
  MemoryManager() = default;
  explicit MemoryManager(llm_engine::BlockManager& blockManager);

  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;
  MemoryManager(MemoryManager&&) = delete;
  MemoryManager& operator=(MemoryManager&&) = delete;

  domain::ManageMemoryResult handle_task(const domain::ManageMemoryTask& task);

 private:
  domain::ManageMemoryStatus allocateKv(const domain::ManageMemoryTask& task,
                                        std::vector<int>& outSlotIds);
  void deallocateKv(const domain::TaskID& taskId, std::vector<int> slotIds);

  llm_engine::BlockManager* blockManager = nullptr;
};

}  // namespace tt::services
