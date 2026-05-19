// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>

#include "services/memory_services/memory_manager.hpp"

namespace tt::services {

class BlazeMemoryManager : public MemoryManager {
 public:
  BlazeMemoryManager() = default;
  ~BlazeMemoryManager() = default;
  std::optional<domain::ManageMemoryTask> getRequest() override;
  void replyAllocateSuccess(uint32_t taskId, uint32_t slotId);
  void replyAllocateFailure(uint32_t taskId);
};

}  // namespace tt::services
