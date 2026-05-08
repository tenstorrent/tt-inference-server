// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "tt_llm_engine/scheduler/decode/decode_scheduler.hpp"
#include "services/memory_services/memory_manager.hpp"

namespace tt::services {

class BlazeMemoryManager : public MemoryManager {
  using onEvictCb = std::function<void(uint32_t slotId)>;

 public:
  BlazeMemoryManager(
      tt_llm_engine::scheduler::decode::DecodeScheduler& decodeScheduler,
      onEvictCb onEvict);
  ~BlazeMemoryManager() = default;

  std::optional<domain::ManageMemoryTask> getRequest() override;

  void handleRequest(const domain::ManageMemoryTask& request) override;

  void handleResponse(uint32_t taskId, uint32_t slotId) override;

 private:
  std::unordered_set<uint32_t> allocating;
  std::unordered_map</*taskId*/ uint32_t, /*slotId*/ uint32_t> cancelling;
  tt_llm_engine::scheduler::decode::DecodeScheduler& decodeScheduler;
  onEvictCb onEvict;
  std::optional<domain::ManageMemoryTask> pendingRetry;
};

}  // namespace tt::services
