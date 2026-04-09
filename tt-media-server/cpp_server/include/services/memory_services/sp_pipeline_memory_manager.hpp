// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <unordered_map>

#include "pipeline_manager/pipeline_manager.hpp"
#include "services/memory_services/async_memory_manager.hpp"

namespace tt::services {

class SpPipelineMemoryManager : public AsyncMemoryManager {
  using onEvictCb = std::function<void(uint32_t slotId)>;

 public:
  SpPipelineMemoryManager(
      ::pipeline_manager::PipelineManager& pipelineManager,
      onEvictCb onEvict);
  ~SpPipelineMemoryManager() = default;

  void handleRequest(const domain::ManageMemoryTask& request) override;

  void handleResponse(uint32_t requestId, uint32_t slotId) override;

 private:
  ::pipeline_manager::PipelineManager& pipelineManager;
  std::unordered_map<uint32_t, uint32_t> allocating;
  uint32_t nextRequestID{0};
  onEvictCb onEvict;
};

}  // namespace tt::services