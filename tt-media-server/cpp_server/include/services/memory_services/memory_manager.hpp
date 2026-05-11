// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <optional>

#include "domain/manage_memory.hpp"
#include "ipc/boost/memory_queue.hpp"

namespace tt::services {

class MemoryManager {
 public:
  MemoryManager();
  virtual ~MemoryManager();

  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;
  MemoryManager(MemoryManager&&) = delete;
  MemoryManager& operator=(MemoryManager&&) = delete;

  virtual std::optional<domain::ManageMemoryTask> getRequest();
  virtual void handleRequest(const domain::ManageMemoryTask& request) = 0;

  // Optional method for asynchronous memory managers that receive responses
  // from an external system. Synchronous managers don't
  // need to override this. Async managers override to
  // complete allocation after receiving a response.
  virtual void handleResponse(uint32_t requestId, uint32_t slotId) {
    // Default implementation does nothing - only async managers need this
    (void)requestId;
    (void)slotId;
  }

 protected:
  std::unique_ptr<ipc::boost::MemoryRequestQueue> requestQueue;
  std::unique_ptr<ipc::boost::MemoryResultQueue> resultQueue;
};

}  // namespace tt::services
