// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "config/settings.hpp"
#include "domain/llm/sequence.hpp"
#include "pipeline_manager/pipeline_manager_types.hpp"
#include "utils/logger.hpp"

namespace tt::runners::blaze_utils {

using namespace tt::domain::llm;

namespace pm = tt_blaze::pipeline_manager;

inline pm::ISRequest makeAllocateRequest(uint32_t requestId) {
  return {
      .type = pm::RequestType::ALLOCATE, .request_id = requestId, .tokens = {}};
}

inline pm::ISRequest makeCancelRequest(uint32_t requestId, uint32_t slotId) {
  return {.type = pm::RequestType::CANCEL,
          .request_id = requestId,
          .slot_id = slotId,
          .tokens = {}};
}

inline pm::ISRequest makeEvictRequest(uint32_t requestId, uint32_t slotId) {
  return {.type = pm::RequestType::DEALLOCATE,
          .request_id = requestId,
          .slot_id = slotId,
          .tokens = {}};
}

inline pm::GenerationParams makeGenerationParams(
    const tt::domain::llm::Sequence& seq) {
  return {
      .max_new_tokens =
          static_cast<uint32_t>(seq.getSamplingParams().max_tokens.value_or(
              static_cast<int>(tt::config::maxContextLength()))),
      .spec_decode = seq.getSamplingParams().fast_mode,
      .ignore_eos = seq.getSamplingParams().ignore_eos,
      .temperature = seq.getSamplingParams().temperature,
      .top_p = seq.getSamplingParams().top_p.value_or(1.0f),
      .top_k = static_cast<int32_t>(seq.getSamplingParams().top_k.value_or(-1)),
      .disaggregated_decode = seq.isDisaggregated()};
}

inline void fillSequenceFields(pm::ISRequest& req,
                               const tt::domain::llm::Sequence& seq) {
  req.tokens.assign(seq.getTokenIds().begin(), seq.getTokenIds().end());
  req.gen = makeGenerationParams(seq);
}

inline pm::ISRequest makeSubmitRequest(uint32_t slotId,
                                       const tt::domain::llm::Sequence& seq) {
  pm::ISRequest req{};
  req.type = pm::RequestType::SUBMIT;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

inline pm::ISRequest makeContinueRequest(uint32_t slotId,
                                         const tt::domain::llm::Sequence& seq) {
  pm::ISRequest req{};
  req.type = pm::RequestType::CONTINUE;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

struct SlotContext {
  uint32_t taskId;
  bool ignoreEos;
  uint32_t specAcceptsAtStart = 0;
  uint32_t specRejectsAtStart = 0;
  uint32_t tokensGenerated = 0;
};

class SlotIndex {
 public:
  void assign(uint32_t slotId, SlotContext ctx) {
    auto existingSlot = bySlot.find(slotId);
    if (existingSlot != bySlot.end() &&
        existingSlot->second.taskId != ctx.taskId) {
      byTask.erase(existingSlot->second.taskId);
    }
    auto existingTask = byTask.find(ctx.taskId);
    if (existingTask != byTask.end() && existingTask->second != slotId) {
      bySlot.erase(existingTask->second);
    }
    byTask.insert_or_assign(ctx.taskId, slotId);
    bySlot.insert_or_assign(slotId, std::move(ctx));
  }

  std::optional<uint32_t> eraseBySlot(uint32_t slotId) {
    auto it = bySlot.find(slotId);
    if (it == bySlot.end()) {
      return std::nullopt;
    }
    auto taskId = it->second.taskId;
    byTask.erase(taskId);
    bySlot.erase(it);
    return taskId;
  }

  bool isTaskRunning(uint32_t taskId) const {
    return byTask.find(taskId) != byTask.end();
  }

  SlotContext* contextBySlot(uint32_t slotId) {
    auto it = bySlot.find(slotId);
    return it != bySlot.end() ? &it->second : nullptr;
  }

  uint32_t slotIdByTask(uint32_t taskId) const {
    auto it = byTask.find(taskId);
    return it != byTask.end() ? it->second : tt::domain::INVALID_SLOT_ID;
  }

  size_t size() const { return bySlot.size(); }
  bool empty() const { return bySlot.empty(); }

 private:
  std::unordered_map<uint32_t, SlotContext> bySlot;
  std::unordered_map<uint32_t, uint32_t> byTask;
};

struct CancelTombstones {
  std::deque<uint32_t> cancelTombstoneOrder;
  std::unordered_set<uint32_t> cancelTombstoneSet;
  const size_t MAX_CANCEL_TOMBSTONES = tt::config::pmMaxUsers() * 2;
  bool consumeCancelTombstone(uint32_t taskId) {
    if (cancelTombstoneSet.erase(taskId) == 0) {
      return false;
    }
    // Lazy removal from the deque: leave a stale entry behind. It will be
    // skipped when popped during eviction. This keeps the common path O(1).
    return true;
  }
  void rememberCancelTombstone(uint32_t taskId) {
    auto shouldRemoveOldest = [&]() {
      return cancelTombstoneOrder.size() > MAX_CANCEL_TOMBSTONES;
    };
    if (!cancelTombstoneSet.insert(taskId).second) {
      return;
    }
    cancelTombstoneOrder.push_back(taskId);
    while (shouldRemoveOldest()) {
      TT_LOG_DEBUG(
          "[CancelTombstones] rememberCancelTombstone: removing oldest "
          "cancel tombstone because we have too many: {}",
          cancelTombstoneOrder.front());
      uint32_t oldest = cancelTombstoneOrder.front();
      cancelTombstoneOrder.pop_front();
      cancelTombstoneSet.erase(oldest);
    }
  }
};

}  // namespace tt::runners::blaze_utils
