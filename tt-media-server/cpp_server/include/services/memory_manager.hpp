// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <string>
#include <vector>

#include "domain/manage_memory.hpp"
#include "utils/concurrent_map.hpp"

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
  struct Reservation {
    domain::KvMemoryLayout layout{domain::KvMemoryLayout::Paged};
    std::vector<int> blockIds;
    std::vector<domain::KvDestination> locations;
  };

  domain::ManageMemoryStatus allocateKv(
      std::int32_t inputSeqLen, std::vector<int>& outBlockIds,
      std::vector<domain::KvDestination>& outLocations);
  void deallocateKv(const std::vector<int>& blockIds);

  llm_engine::BlockManager* blockManager = nullptr;
  ConcurrentMap<std::string, Reservation> reservations;
};

}  // namespace tt::services
