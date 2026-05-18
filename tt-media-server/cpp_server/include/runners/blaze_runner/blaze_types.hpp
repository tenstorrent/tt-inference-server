// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "domain/llm/sequence.hpp"
#include "tt_llm_engine/scheduler/decode/decode_types.hpp"

namespace tt::runners::blaze_types {
namespace ds = tt_llm_engine::scheduler::decode;

enum class SlotState {
  FREE,
  AWAITING_ALLOCATE_ACK,
  IDLE,       // allocated, no request running (slot retained for prefix cache)
  SUBMITTED,  // SUBMIT/CONTINUE in flight, tokens flowing
  AWAITING_STOP_ACK,   // STOP in flight; deferred actions latched
  AWAITING_EVICT_ACK,  // CANCEL/EVICT in flight; terminal-ish
};

struct SlotContext {
  SlotState state = SlotState::FREE;
  std::optional<uint32_t> taskId;
  bool ignoreEos = false;
  uint32_t specAcceptsAtStart = 0;
  uint32_t specRejectsAtStart = 0;
  uint32_t tokensGenerated = 0;
  std::optional<uint32_t> pendingAckRequestId = std::nullopt;
  std::optional<ds::ISRequest> deferredEvict = std::nullopt;
  std::optional<tt::domain::llm::Sequence> deferredSubmit = std::nullopt;
};

class SlotManager {
 public:
  SlotManager(int numSlots) { slots.resize(numSlots); }
  ~SlotManager() = default;

  SlotContext& getSlotContext(uint32_t slotId) { return slots[slotId]; }

  void clearSlotContext(uint32_t slotId) {
    if (slots[slotId].taskId.has_value()) {
      taskToSlot.erase(slots[slotId].taskId.value());
    }
    slots[slotId] = SlotContext{};
  }
  void clearAllSlotContexts() {
    for (auto& slot : slots) {
      if (slot.taskId.has_value()) {
        taskToSlot.erase(slot.taskId.value());
      }
      slot = SlotContext{};
    }
  }

  SlotContext* getSlotContextByTaskId(uint32_t taskId) {
    auto it = taskToSlot.find(taskId);
    if (it == taskToSlot.end()) {
      return nullptr;
    }
    return &getSlotContext(it->second);
  }

 private:
  std::vector<SlotContext> slots;
  std::unordered_map</*taskId*/ uint32_t, /*slotId*/ uint32_t> taskToSlot;
};

}  // namespace tt::runners::blaze_types