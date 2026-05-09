// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <deque>
#include <unordered_set>

#include "config/runner_config.hpp"
#include "domain/llm/sequence.hpp"
#include "pipeline_manager/pipeline_manager_types.hpp"

namespace tt::runners::blaze_utils {

using namespace tt::domain::llm;

namespace pm = tt_blaze::pipeline_manager;

inline pm::ISRequest makeAllocateRequest(uint32_t requestId) {
  return {
      .type = pm::RequestType::ALLOCATE, .request_id = requestId, .tokens = {}};
}

inline pm::ISRequest makeEvictRequest(uint32_t requestId, uint32_t slotId) {
  return {.type = pm::RequestType::CANCEL,
          .request_id = requestId,
          .slot_id = slotId,
          .tokens = {}};
}

inline pm::GenerationParams makeGenerationParams(
    const tt::domain::llm::Sequence& seq) {
  return {
      .max_new_tokens =
          static_cast<uint32_t>(seq.getSamplingParams().max_tokens.value_or(
              static_cast<int>(config::LLMConfig::MAX_INPUT_TOKENS))),
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

struct CancelTombstones {
  std::deque<uint32_t> cancelTombstoneOrder;
  std::unordered_set<uint32_t> cancelTombstoneSet;
  static constexpr size_t MAX_CANCEL_TOMBSTONES = 256;
  bool consumeCancelTombstone(uint32_t taskId) {
    if (cancelTombstoneSet.erase(taskId) == 0) {
      return false;
    }
    // Lazy removal from the deque: leave a stale entry behind. It will be
    // skipped when popped during eviction. This keeps the common path O(1).
    return true;
  }
  void rememberCancelTombstone(uint32_t taskId) {
    if (!cancelTombstoneSet.insert(taskId).second) {
      return;
    }
    cancelTombstoneOrder.push_back(taskId);
    while (cancelTombstoneSet.size() > MAX_CANCEL_TOMBSTONES &&
           !cancelTombstoneOrder.empty()) {
      uint32_t oldest = cancelTombstoneOrder.front();
      cancelTombstoneOrder.pop_front();
      cancelTombstoneSet.erase(oldest);
    }
  }
};

}  // namespace tt::runners::blaze_utils
