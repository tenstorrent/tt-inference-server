// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once
#include <cassert>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "domain/llm/sequence.hpp"
#include "tt_llm_engine/scheduler/decode/decode_types.hpp"
#include "utils/logger.hpp"

namespace tt::runners::blaze_types {
namespace ds = tt_llm_engine::scheduler::decode;

enum class SlotState {
  FREE,
  IDLE,       // allocated, no request running (slot retained for prefix cache)
  SUBMITTED,  // SUBMIT/CONTINUE in flight, tokens flowing
  AWAITING_STOP_ACK,   // STOP in flight; deferred actions latched
  AWAITING_EVICT_ACK,  // CANCEL/EVICT in flight; terminal-ish
};

inline const char* toString(SlotState state) {
  switch (state) {
    case SlotState::FREE: return "FREE";
    case SlotState::IDLE: return "IDLE";
    case SlotState::SUBMITTED: return "SUBMITTED";
    case SlotState::AWAITING_STOP_ACK: return "AWAITING_STOP_ACK";
    case SlotState::AWAITING_EVICT_ACK: return "AWAITING_EVICT_ACK";
  }
}

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
  
  void setState(SlotState newState) {
    if (newState == state) return;   // idempotent self-loops are fine
  
    bool legal = false;
    switch (state) {
      case SlotState::FREE:
        // if we are free, we can only become idle, by allocation
        legal = (newState == SlotState::IDLE);
        break;
      case SlotState::IDLE:
        // if we are idle, we can either submit or evict
        legal = (newState == SlotState::SUBMITTED ||
                 newState == SlotState::AWAITING_EVICT_ACK);
        break;
      case SlotState::SUBMITTED:
        // if we are submitted , we can go to waiting for stop, idle(if we finish)
        // in theory, we can also evict a running sequence, that that is impossible right now
        legal = (newState == SlotState::IDLE ||
                 newState == SlotState::AWAITING_STOP_ACK);
        break;
      case SlotState::AWAITING_STOP_ACK:
        // if we are waiting for stop ack, we can become idle by just stopping naturally
        // we can also jump to awaiting evict ack if we are deferred evict
        // we can also jump to submitted if we are deferred submit/continue
        legal = (newState == SlotState::IDLE ||
                 newState == SlotState::SUBMITTED ||
                 newState == SlotState::AWAITING_EVICT_ACK);
        break;
      case SlotState::AWAITING_EVICT_ACK:
        // if we are waiting for evict ack, we can only become free
        legal = (newState == SlotState::FREE);
        break;
    }
  
    if (!legal) {
      TT_LOG_ERROR("[SlotContext] illegal transition: {} -> {} (taskId={})",
                   toString(state), toString(newState),
                   taskId.has_value() ? std::to_string(*taskId) : "none");
      assert(false && "illegal slot state transition");
    }
    state = newState;
  }
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