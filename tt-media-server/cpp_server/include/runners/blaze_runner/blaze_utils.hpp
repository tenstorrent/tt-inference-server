// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <unordered_map>

#include "config/settings.hpp"
#include "domain/llm/sequence.hpp"
#include "tt_llm_engine/scheduler/decode/decode_types.hpp"

namespace tt::runners::blaze_utils {

using namespace tt::domain::llm;

namespace ds = tt_llm_engine::scheduler::decode;

inline ds::ISRequest makeAllocateRequest(uint32_t requestId) {
  return {
      .type = ds::RequestType::ALLOCATE, .request_id = requestId, .tokens = {}};
}

inline ds::ISRequest makeEvictRequest(uint32_t requestId, uint32_t slotId) {
  return {.type = ds::RequestType::CANCEL,
          .request_id = requestId,
          .slot_id = slotId,
          .tokens = {}};
}

inline ds::ISRequest makeCancelRequest(uint32_t requestId) {
  return {.type = ds::RequestType::CANCEL,
          .request_id = requestId,
          .slot_id = ds::INVALID_SLOT,
          .tokens = {}};
}

inline ds::GenerationParams makeGenerationParams(
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

inline void fillSequenceFields(ds::ISRequest& req,
                               const tt::domain::llm::Sequence& seq) {
  req.tokens.assign(seq.getTokenIds().begin(), seq.getTokenIds().end());
  req.gen = makeGenerationParams(seq);
}

inline ds::ISRequest makeSubmitRequest(uint32_t slotId,
                                       const tt::domain::llm::Sequence& seq) {
  ds::ISRequest req{};
  req.type = ds::RequestType::SUBMIT;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

inline ds::ISRequest makeContinueRequest(uint32_t slotId,
                                         const tt::domain::llm::Sequence& seq) {
  ds::ISRequest req{};
  req.type = ds::RequestType::CONTINUE;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

struct SlotContext {
  uint32_t taskId;
  bool stopped = false;
  bool ignoreEos;
  uint32_t specAcceptsAtStart = 0;
  uint32_t specRejectsAtStart = 0;
  uint32_t tokensGenerated = 0;
};

struct SlotIndex {
  std::optional<SlotContext> getSlotContextByTaskId(uint32_t taskId) {
    if (auto it = slotContextByTaskId.find(taskId);
        it != slotContextByTaskId.end()) {
      return it->second;
    }
    return std::nullopt;
  }
  std::optional<SlotContext> getSlotContextBySlotId(uint32_t slotId) {
    if (auto it = slotContextBySlotId.find(slotId);
        it != slotContextBySlotId.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  void eraseSlotContextByTaskId(uint32_t taskId) {
    slotContextByTaskId.erase(taskId);
  }
  void eraseSlotContextBySlotId(uint32_t slotId) {
    slotContextBySlotId.erase(slotId);
  }
  void addSlotContext(uint32_t slotId, SlotContext context) {
    slotContextByTaskId.insert_or_assign(context.taskId, context);
    slotContextBySlotId.insert_or_assign(slotId, context);
  }

  bool empty() {
    return slotContextByTaskId.empty() && slotContextBySlotId.empty();
  }

  size_t size() { return slotContextByTaskId.size(); }

  std::unordered_map<uint32_t, SlotContext> slotContextByTaskId;
  std::unordered_map<uint32_t, SlotContext> slotContextBySlotId;
};

}  // namespace tt::runners::blaze_utils
