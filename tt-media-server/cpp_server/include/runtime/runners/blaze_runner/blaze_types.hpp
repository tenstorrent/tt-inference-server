// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "domain/llm/sequence.hpp"
#include "domain/manage_memory.hpp"
#include "tt_llm_engine/scheduler/decode/decode_types.hpp"
#include "utils/logger.hpp"

namespace tt::runners::blaze {
namespace ds = tt_llm_engine::scheduler::decode;

enum class SlotState {
  FREE,
  IDLE,     // allocated, no request running (slot retained for prefix cache)
  RUNNING,  // SUBMIT/CONTINUE in flight, tokens flowing
  AWAITING_STOP_ACK,   // STOP in flight; deferred actions latched
  AWAITING_EVICT_ACK,  // EVICT in flight; terminal-ish
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
  std::unique_ptr<tt::domain::llm::Sequence> deferredContinue = nullptr;

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
        // evict
        legal = (newState == SlotState::IDLE ||
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
      throw std::runtime_error("illegal slot state transition");
    }
    state = newState;
  }
};
// These requests are pending due to scheduler queue full and need to be
// retried on the next step.
struct PendingRequests {
  std::unique_ptr<tt::domain::llm::Sequence> pendingTask;
  std::optional<tt::domain::ManageMemoryTask> pendingMemoryTask;
  std::optional<uint32_t> pendingCancelTaskId;
};

}  // namespace tt::runners::blaze