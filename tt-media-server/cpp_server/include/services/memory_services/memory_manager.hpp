// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <optional>

#include "domain/manage_memory.hpp"
#include "ipc/boost/boost_memory_queue.hpp"

namespace tt::services {

// IPC adapter for memory-management messages exchanged with SessionManager.
// Owns the boost queues; provides pull (getRequest) and push (replyAllocate*)
// primitives. Holds no state about what the runner does with the slots/blocks
// it allocates — that's the runner's concern.
class MemoryManager {
 public:
  MemoryManager();
  ~MemoryManager();

  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;
  MemoryManager(MemoryManager&&) = delete;
  MemoryManager& operator=(MemoryManager&&) = delete;

  std::optional<domain::ManageMemoryTask> getRequest();

  // slotId is an opaque handle from the runner's perspective — for blaze it's
  // a kv-cache slot index, for future paged runners it may be a block handle,
  // etc. SessionManager just stores it and echoes it back on the next request.
  void replyAllocateSuccess(uint32_t taskId, uint32_t slotId);
  void replyAllocateFailure(uint32_t taskId);

 private:
  std::unique_ptr<ipc::boost::MemoryRequestQueue> requestQueue;
  std::unique_ptr<ipc::boost::MemoryResultQueue> resultQueue;
};

}  // namespace tt::services
