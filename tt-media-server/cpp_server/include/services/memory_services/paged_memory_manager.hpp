// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "services/memory_services/memory_manager.hpp"

namespace llm_engine {
class BlockManager;
}

namespace tt::services {

class PagedMemoryManager : public MemoryManager {
 public:
  explicit PagedMemoryManager(llm_engine::BlockManager& blockManager);

  void handleRequest(const domain::ManageMemoryTask& request) override;

 private:
  domain::ManageMemoryStatus allocateKv(const domain::ManageMemoryTask& task,
                                        std::vector<int>& outSlotIds);
  void deallocateKv(uint32_t taskId, std::vector<int> slotIds);

  llm_engine::BlockManager* blockManager;
};

}  // namespace tt::services
