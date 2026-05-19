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

namespace tt::runners::blaze_slot_manager {
namespace ds = tt_llm_engine::scheduler::decode;

enum class SlotState {
  FREE,
  IDLE,     // allocated, no request running (slot retained for prefix cache)
  RUNNING,  // SUBMIT/CONTINUE in flight, tokens flowing
  AWAITING_STOP_ACK,   // STOP in flight; deferred actions latched
  AWAITING_EVICT_ACK,  // CANCEL/EVICT in flight; terminal-ish
};

inline const char* toString(SlotState state) {
  switch (state) {
    case SlotState::FREE:
      return "FREE";
    case SlotState::IDLE:
      return "IDLE";
    case SlotState::RUNNING:
    return "RUNNING";
    case SlotState::AWAITING_STOP_ACK:
      return "AWAITING_STOP_ACK";
    case SlotState::AWAITING_EVICT_ACK:
      return "AWAITING_EVICT_ACK";
  }
  return "UNKNOWN";
}

struct SlotContext {
  SlotState state = SlotState::FREE;
  std::optional<uint32_t> taskId;
  uint32_t slotId;
  bool ignoreEos = false;
  uint32_t specAcceptsAtStart = 0;
  uint32_t specRejectsAtStart = 0;
  uint32_t tokensGenerated = 0;
  std::optional<uint32_t> pendingAckRequestId = std::nullopt;
  std::optional<ds::ISRequest> deferredEvict = std::nullopt;
  std::unique_ptr<tt::domain::llm::Sequence> deferredSubmit = nullptr;

  void setState(SlotState newState) {
    if (newState == state) return;

    bool legal = false;
    switch (state) {
      case SlotState::FREE:
        // if we are free, we can only become idle, by allocation
        legal = (newState == SlotState::IDLE);
        break;
      case SlotState::IDLE:
        // if we are idle, we can either submit or evict
        legal = (newState == SlotState::RUNNING ||
                 newState == SlotState::AWAITING_EVICT_ACK);
        break;
      case SlotState::RUNNING:
        // if we are submitted , we can go to waiting for stop, idle or awaiting
        // evict ack
        legal = (newState == SlotState::IDLE ||
                 newState == SlotState::AWAITING_STOP_ACK ||
                 newState == SlotState::AWAITING_EVICT_ACK);
        break;
      case SlotState::AWAITING_STOP_ACK:
        // if we are waiting for stop ack, we can become idle by just stopping
        // naturally we can also jump to awaiting evict ack if we are deferred
        // evict we can also jump to submitted if we are deferred
        // submit/continue
        legal =
            (newState == SlotState::IDLE || newState == SlotState::RUNNING ||
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
  SlotManager(int numSlots) {
    slots.resize(numSlots);
    for (size_t i = 0; i < slots.size(); ++i) {
      slots[i].slotId = i;
    }
  }
  ~SlotManager() = default;

  SlotContext& getSlotContext(uint32_t slotId) { return slots[slotId]; }

  void clearSlotContext(uint32_t slotId) {
    auto& slot = slots[slotId];
    if (slot.taskId) taskToSlot.erase(*slot.taskId);
    slot.taskId.reset();
    slot.ignoreEos = false;
    slot.specAcceptsAtStart = 0;
    slot.specRejectsAtStart = 0;
    slot.tokensGenerated = 0;
    slot.pendingAckRequestId.reset();
    slot.deferredEvict.reset();
    slot.deferredSubmit.reset();
    setSlotState(slotId, SlotState::FREE);
  }
  void clearAllSlotContexts() {
    for (size_t i = 0; i < slots.size(); ++i) {
      clearSlotContext(i);
    }
  }

  SlotContext* getSlotContextByTaskId(uint32_t taskId) {
    auto it = taskToSlot.find(taskId);
    if (it == taskToSlot.end()) {
      return nullptr;
    }
    return &getSlotContext(it->second);
  }

  void bindTaskToSlot(uint32_t taskId, uint32_t slotId) {
    if (taskToSlot.count(taskId) > 0) {
      TT_LOG_ERROR("[SlotManager] taskId={} already bound to slotId={}", taskId,
                   taskToSlot[taskId]);
      assert(false && "taskId already bound to slotId");
    }
    taskToSlot[taskId] = slotId;
  }

  void unbindTaskFromSlot(uint32_t taskId) {
    auto it = taskToSlot.find(taskId);
    if (it == taskToSlot.end()) {
      TT_LOG_ERROR("[SlotManager] taskId={} not bound to any slot", taskId);
      assert(false && "taskId not bound to any slot");
    }
    taskToSlot.erase(it);
  }

  uint32_t activeRunningCount() const { return runningCount; }

  void setSlotState(uint32_t slotId, SlotState state) {
    if (state == SlotState::RUNNING) runningCount++;
    if (slots[slotId].state == SlotState::RUNNING) runningCount--;
    slots[slotId].setState(state);
  }

  void setSlotAsIdle(uint32_t slotId) {
    auto& slotContext = slots[slotId];
    if (slotContext.taskId) taskToSlot.erase(*slotContext.taskId);
    slotContext.taskId.reset();
    slotContext.ignoreEos = false;
    slotContext.specAcceptsAtStart = 0;
    slotContext.specRejectsAtStart = 0;
    slotContext.tokensGenerated = 0;
    slotContext.pendingAckRequestId.reset();
    setSlotState(slotId, SlotState::IDLE);
  }

 private:
  std::vector<SlotContext> slots;
  std::unordered_map</*taskId*/ uint32_t, /*slotId*/ uint32_t> taskToSlot;
  uint32_t runningCount = 0;
};

}  // namespace tt::runners::blaze_types