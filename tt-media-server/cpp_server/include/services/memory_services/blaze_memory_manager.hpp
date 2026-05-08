// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include "pipeline_manager/pipeline_manager.hpp"
#include "services/memory_services/memory_manager.hpp"

namespace tt::services {

class BlazeMemoryManager : public MemoryManager {
  using onEvictCb = std::function<void(uint32_t slotId)>;

 public:
  BlazeMemoryManager(
      tt_blaze::pipeline_manager::PipelineManager& pipelineManager,
      onEvictCb onEvict);
  ~BlazeMemoryManager() = default;

  std::optional<domain::ManageMemoryTask> getRequest() override;

  void handleRequest(const domain::ManageMemoryTask& request) override;

  void handleResponse(uint32_t requestId, uint32_t slotId) override;

 private:
  tt_blaze::pipeline_manager::PipelineManager& pipelineManager;
  std::unordered_map<uint32_t, uint32_t> allocating;
  std::unordered_map<uint32_t, uint32_t> cancelling;
  uint32_t nextRequestID{0};
  onEvictCb onEvict;
  std::optional<domain::ManageMemoryTask> pendingRetry;
};

}  // namespace tt::services
