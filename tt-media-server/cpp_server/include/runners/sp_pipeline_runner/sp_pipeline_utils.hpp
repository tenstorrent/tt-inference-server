// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "config/runner_config.hpp"
#include "llm_runner/sequence.hpp"
#include "pipeline_manager/pipeline_manager_types.hpp"

namespace tt::runners::sp_pipeline_utils {

namespace pm = tt_blaze::pipeline_manager;

inline pm::ISRequest makeAllocateRequest(uint32_t requestId) {
  return {.type = pm::RequestType::ALLOCATE, .request_id = requestId};
}

inline pm::ISRequest makeCancelRequest(uint32_t slotId) {
  return {.type = pm::RequestType::CANCEL, .slot_id = slotId};
}

inline pm::GenerationParams makeGenerationParams(
    const llm_engine::Sequence& seq) {
  return {
      .max_new_tokens =
          static_cast<uint32_t>(seq.samplingParams->max_tokens.value_or(
              static_cast<int>(config::LLMConfig::MAX_INPUT_TOKENS))),
      .spec_decode = seq.samplingParams->fast_mode,
      .temperature = seq.samplingParams->temperature,
      .top_p = seq.samplingParams->top_p.value_or(1.0f),
      .top_k = static_cast<int32_t>(seq.samplingParams->top_k.value_or(-1))};
}

inline void fillSequenceFields(pm::ISRequest& req,
                               const llm_engine::Sequence& seq) {
  req.tokens.assign(seq.tokenIds.begin(), seq.tokenIds.end());
  req.gen = makeGenerationParams(seq);
}

inline pm::ISRequest makeSubmitRequest(uint32_t slotId,
                                       const llm_engine::Sequence& seq) {
  pm::ISRequest req{};
  req.type = pm::RequestType::SUBMIT;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

inline pm::ISRequest makeContinueRequest(uint32_t slotId,
                                         const llm_engine::Sequence& seq) {
  pm::ISRequest req{};
  req.type = pm::RequestType::CONTINUE;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

}  // namespace tt::runners::sp_pipeline_utils