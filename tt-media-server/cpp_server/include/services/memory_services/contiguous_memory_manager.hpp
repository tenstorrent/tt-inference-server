// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "services/memory_services/memory_manager.hpp"

namespace tt::services {

class ContiguousMemoryManager : public MemoryManager {
 public:
  ContiguousMemoryManager() = default;

  void handleRequest(const domain::ManageMemoryTask& request) override;
};

}  // namespace tt::services
