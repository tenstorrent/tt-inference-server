// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once
#include <cassert>
#include <chrono>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "runtime/runners/blaze_runner/blaze_types.hpp"
#include "utils/logger.hpp"
namespace tt::runners::blaze {
class SlotManager {
 public:
  SlotManager(size_t numSlots) {
    slots.resize(numSlots);
    for (size_t i = 0; i < slots.size(); ++i) {
      slots[i].slotId = i;
    }
  }
  ~SlotManager() = default;

  SlotContext& getSlotContext(uint32_t slotId) { return slots[slotId]; }

  std::string dumpSlotStates() const {
    std::stringstream ss;
    for (size_t i = 0; i < slots.size(); ++i) {
      ss << "Slot " << i << ": " << toString(slots[i].state)
         << " (taskId=" << slots[i].taskId.value_or(-1) << ")\n";
      ss << " pendingAckRequestId=" << slots[i].pendingAckRequestId.value_or(-1)
         << "\n";
      ss << " deferredEvict="
         << (slots[i].deferredEvict.has_value() ? "true" : "false") << "\n";
      ss << " deferredContinue="
         << (slots[i].deferredContinue ? "true" : "false") << "\n";
      ss << " currentPosition=" << slots[i].currentPosition << "\n";
      ss << " ignoreEos=" << slots[i].ignoreEos << "\n";
      ss << " specAcceptsAtStart=" << slots[i].specAcceptsAtStart << "\n";
      ss << " specRejectsAtStart=" << slots[i].specRejectsAtStart << "\n";
      ss << " tokensGenerated=" << slots[i].tokensGenerated << "\n";
      ss << "----------------------------------------\n";
    }
    return ss.str();
  }

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
    slot.deferredContinue.reset();
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
    slots[slotId].lastProgressTime = std::chrono::steady_clock::now();
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

  const std::vector<SlotContext>& getSlots() const { return slots; }

 private:
  std::vector<SlotContext> slots;
  std::unordered_map</*taskId*/ uint32_t, /*slotId*/ uint32_t> taskToSlot;
  uint32_t runningCount = 0;
};

}  // namespace tt::runners::blaze
