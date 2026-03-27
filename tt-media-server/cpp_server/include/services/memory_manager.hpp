// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "domain/manage_memory.hpp"
#include "utils/concurrent_map.hpp"

namespace tt::services {

class MemoryManager {
 public:
  MemoryManager() = default;

  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;
  MemoryManager(MemoryManager&&) = delete;
  MemoryManager& operator=(MemoryManager&&) = delete;

  domain::ManageMemoryResult handle_task(const domain::ManageMemoryTask& task);

 private:
  struct Reservation {
    domain::KvMemoryLayout layout{domain::KvMemoryLayout::Paged};
    std::vector<domain::KvDestination> locations;
  };

  ConcurrentMap<uint32_t, Reservation> reservations;
};

}  // namespace tt::services
