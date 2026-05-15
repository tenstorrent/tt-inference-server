// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>

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

inline ds::ISRequest makeStopRequest(uint32_t requestId, uint32_t slotId) {
  return {.type = ds::RequestType::STOP,
          .request_id = requestId,
          .slot_id = slotId,
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
  bool ignoreEos;
  uint32_t specAcceptsAtStart = 0;
  uint32_t specRejectsAtStart = 0;
  uint32_t tokensGenerated = 0;
};

// Buffers a deferred SUBMIT/CONTINUE for a slot that currently has an
// in-flight STOP. The matching STOP ack carries `request_id ==
// expectedStopRequestId` (the cancelled sequence's taskId); once it arrives
// the deferred `sequence` (if any) is re-submitted via handleRequest.
struct PendingSubmit {
  uint32_t expectedStopRequestId;
  std::unique_ptr<tt::domain::llm::Sequence> sequence;
};

}  // namespace tt::runners::blaze_utils
