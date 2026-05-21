// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once
#include <cassert>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "runtime/runners/blaze_runner/blaze_types.hpp"
#include "utils/logger.hpp"
namespace tt::runners::blaze {
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
    if (!taskToSlot.contains(taskId)) {
      return nullptr;
    }
    return &getSlotContext(taskToSlot[taskId]);
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
    if (!taskToSlot.contains(taskId)) {
      TT_LOG_ERROR("[SlotManager] taskId={} not bound to any slot", taskId);
      assert(false && "taskId not bound to any slot");
    }
    taskToSlot.erase(taskId);
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

}  // namespace tt::runners::blaze