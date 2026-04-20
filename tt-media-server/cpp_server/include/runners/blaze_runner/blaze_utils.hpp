// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "config/types.hpp"
#include "llm_runner/sequence.hpp"
#include "pipeline_manager/pipeline_manager_types.hpp"

namespace tt::runners::blaze_utils {

namespace pm = tt_blaze::pipeline_manager;

inline pm::ISRequest makeAllocateRequest(uint32_t requestId) {
  return {
      .type = pm::RequestType::ALLOCATE, .request_id = requestId, .tokens = {}};
}

inline pm::ISRequest makeCancelRequest(uint32_t slotId) {
  return {.type = pm::RequestType::CANCEL, .slot_id = slotId, .tokens = {}};
}

inline pm::GenerationParams makeGenerationParams(
    const tt::runners::llm_engine::Sequence& seq) {
  return {.max_new_tokens =
              static_cast<uint32_t>(seq.getSamplingParams().max_tokens.value_or(
                  static_cast<int>(config::LLMConfig::MAX_INPUT_TOKENS))),
          .spec_decode = seq.getSamplingParams().fast_mode,
          .ignore_eos = seq.getSamplingParams().ignore_eos,
          .temperature = seq.getSamplingParams().temperature,
          .top_p = seq.getSamplingParams().top_p.value_or(1.0f),
          .top_k =
              static_cast<int32_t>(seq.getSamplingParams().top_k.value_or(-1))};
}

inline void fillSequenceFields(pm::ISRequest& req,
                               const tt::runners::llm_engine::Sequence& seq) {
  req.tokens.assign(seq.getTokenIds().begin(), seq.getTokenIds().end());
  req.gen = makeGenerationParams(seq);
}

inline pm::ISRequest makeSubmitRequest(
    uint32_t slotId, const tt::runners::llm_engine::Sequence& seq) {
  pm::ISRequest req{};
  req.type = pm::RequestType::SUBMIT;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

inline pm::ISRequest makeContinueRequest(
    uint32_t slotId, const tt::runners::llm_engine::Sequence& seq) {
  pm::ISRequest req{};
  req.type = pm::RequestType::CONTINUE;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

inline pm::PipelineConfig makePipelineConfig() {
  auto modelRunnerType = tt::config::llmEngineConfig().runner_type;
  switch (modelRunnerType) {
    case tt::config::ModelRunnerType::PIPELINE:
      return pm::SocketConfig{
          .h2d_socket_id = tt::config::blazeSocketDescriptorPrefix() + "_h2d",
          .d2h_socket_id = tt::config::blazeSocketDescriptorPrefix() + "_d2h",
          .connect_timeout_ms = tt::config::pmConnectTimeoutMs(),
          .use_deepseek_md_format = tt::config::useDeepseekMdFormat(),
      };
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return pm::PipelineSimulatorConfig{
          .num_stages = 64,
          .stage_duration_us = 44,
          .decode_token_id = 220,
      };
    default:
      throw std::invalid_argument("Invalid model runner type");
  }
}

}  // namespace tt::runners::blaze_utils
