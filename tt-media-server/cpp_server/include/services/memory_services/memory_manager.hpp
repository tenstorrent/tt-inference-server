// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <optional>

#include "domain/manage_memory.hpp"
#include "ipc/interface/memory_queue.hpp"

namespace tt::services {
class MemoryManager {
 public:
  MemoryManager();
  MemoryManager(std::shared_ptr<ipc::IMemoryRequestQueue> requestQueue,
                std::shared_ptr<ipc::IMemoryResultQueue> resultQueue);
  ~MemoryManager();

  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;
  MemoryManager(MemoryManager&&) = delete;
  MemoryManager& operator=(MemoryManager&&) = delete;

  std::optional<domain::ManageMemoryTask> getRequest();

  void replyAllocateSuccess(uint32_t taskId, uint32_t slotId);
  void replyAllocateFailure(uint32_t taskId);

 private:
  std::shared_ptr<ipc::IMemoryRequestQueue> requestQueue;
  std::shared_ptr<ipc::IMemoryResultQueue> resultQueue;
};

}  // namespace tt::services
