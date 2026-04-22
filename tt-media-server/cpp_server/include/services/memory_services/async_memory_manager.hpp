// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "services/memory_services/memory_manager.hpp"

namespace tt::services {

// Sub-interface for memory managers with asynchronous allocation.
// Synchronous managers (PagedMemoryManager, ContiguousMemoryManager) complete
// allocation inside handleRequest. Async managers send a request to an external
// system and receive the result later via handleResponse.
class AsyncMemoryManager : public MemoryManager {
 public:
  virtual void handleResponse(uint32_t requestId, uint32_t slotId) = 0;
};

}  // namespace tt::services
