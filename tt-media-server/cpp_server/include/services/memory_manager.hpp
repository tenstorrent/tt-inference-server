// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "domain/manage_memory.hpp"

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

  std::mutex reservationMutex;
  std::unordered_map<std::string, Reservation> reservations;
};

}  // namespace tt::services
