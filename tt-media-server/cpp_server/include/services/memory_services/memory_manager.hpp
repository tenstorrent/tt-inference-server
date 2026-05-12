// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <optional>

#include "domain/manage_memory.hpp"
#include "ipc/boost_ipc_queue.hpp"

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

  // Default implementation: ALLOCATE returns SUCCESS with an opaque slotId,
  // EVICT is a no-op. The slotId value is not interpreted by the legacy
  // LLM scheduler (it manages its own block table), so the default acts as a
  // pure session-creation gate. Async managers (e.g. BlazeMemoryManager)
  // override to talk to an external scheduler.
  virtual void handleRequest(const domain::ManageMemoryTask& request);

  // Optional method for asynchronous memory managers that receive responses
  // from an external system. Synchronous managers don't
  // need to override this. Async managers override to
  // complete allocation after receiving a response.
  virtual void handleResponse(uint32_t taskId, uint32_t slotId) {
    // Default implementation does nothing - only async managers need this
    (void)taskId;
    (void)slotId;
  }

 protected:
  std::unique_ptr<ipc::MemoryRequestQueue> requestQueue;
  std::unique_ptr<ipc::MemoryResultQueue> resultQueue;
};

}  // namespace tt::services
