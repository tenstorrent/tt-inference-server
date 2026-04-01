// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <optional>

#include "domain/manage_memory.hpp"
#include "ipc/boost_ipc_memory_queue.hpp"

namespace tt::services {

class MemoryManager {
 public:
  MemoryManager();
  virtual ~MemoryManager();

  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;
  MemoryManager(MemoryManager&&) = delete;
  MemoryManager& operator=(MemoryManager&&) = delete;

  std::optional<domain::ManageMemoryTask> getRequest();
  virtual void handleRequest(const domain::ManageMemoryTask& request) = 0;

 protected:
  std::unique_ptr<ipc::MemoryRequestQueue> requestQueue;
  std::unique_ptr<ipc::MemoryResultQueue> resultQueue;
};

}  // namespace tt::services
