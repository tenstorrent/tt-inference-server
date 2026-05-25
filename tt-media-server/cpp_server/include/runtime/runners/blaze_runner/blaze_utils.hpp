// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>

#include "config/settings.hpp"
#include "domain/llm/sequence.hpp"
#include "runtime/runners/blaze_runner/blaze_slot_manager.hpp"
#include "runtime/runners/blaze_runner/blaze_types.hpp"
#include "tt_llm_engine/scheduler/decode/decode_scheduler.hpp"
#include "tt_llm_engine/scheduler/decode/decode_types.hpp"
#include "utils/logger.hpp"

namespace tt::runners::blaze::utils {

namespace ds = tt_llm_engine::scheduler::decode;

inline ds::ISRequest makeAllocateRequest(uint32_t requestId) {
  return {.type = ds::RequestType::ALLOCATE,
          .request_id = requestId,
          .tokens = {},
          .gen = {}};
}

inline ds::ISRequest makeEvictRequest(uint32_t requestId, uint32_t slotId) {
  return {.type = ds::RequestType::CANCEL,
          .request_id = requestId,
          .slot_id = slotId,
          .tokens = {},
          .gen = {}};
}

inline ds::ISRequest makeStopRequest(uint32_t requestId, uint32_t slotId) {
  return {.type = ds::RequestType::STOP,
          .request_id = requestId,
          .slot_id = slotId,
          .tokens = {},
          .gen = {}};
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
      .disaggregated_decode = seq.isDisaggregated(),
      .position_id = seq.getKVCacheOffset()};
}

inline void fillSequenceFields(ds::ISRequest& req,
                               const tt::domain::llm::Sequence& seq) {
  if (seq.isDisaggregated()) {
    req.tokens = {static_cast<uint32_t>(seq.getLastToken())};
  } else {
    req.tokens.assign(seq.getTokenIds().begin(), seq.getTokenIds().end());
  }
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

// Populates per-run fields on `slot` from `seq`. Snapshots the slot's spec
// counters at this moment so handleOutput can later report per-turn deltas.
// Does not touch state machine / metrics / task binding — caller's job.
inline void initSlotForRun(SlotContext& slot,
                           const tt::domain::llm::Sequence& seq,
                           ds::DecodeScheduler& sched) {
  slot.ignoreEos = seq.getSamplingParams().ignore_eos;
  slot.specAcceptsAtStart = sched.get_spec_accepts(slot.slotId);
  slot.specRejectsAtStart = sched.get_spec_rejects(slot.slotId);
  slot.taskId = seq.taskId;
  slot.tokensGenerated = 0;
}

struct SpecDelta {
  uint32_t accepts;
  uint32_t rejects;
};

// Computes the (accepts, rejects) deltas relative to slot start and logs the
// per-turn acceptance summary.
inline SpecDelta computeAndLogSpecDelta(ds::DecodeScheduler& sched,
                                        const SlotContext& slot,
                                        const ds::OutputMessage& output,
                                        uint32_t taskId, bool hitStop) {
  SpecDelta d{
      .accepts =
          sched.get_spec_accepts(output.slot_id) - slot.specAcceptsAtStart,
      .rejects =
          sched.get_spec_rejects(output.slot_id) - slot.specRejectsAtStart,
  };
  uint32_t total = d.accepts + d.rejects;
  double acceptRate = total > 0 ? 100.0 * d.accepts / total : 0.0;
  TT_LOG_INFO(
      "slot {} turn: accepts={}/{} rate={:.1f}% taskId={} token_id={} "
      "is_complete={} ignoreEos={} hitStop={} tokensGenerated={}",
      output.slot_id, d.accepts, total, acceptRate, taskId, output.token_id,
      output.is_complete, slot.ignoreEos, hitStop, slot.tokensGenerated);
  return d;
}

namespace pl = tt_llm_engine::pipeline;

inline pl::PipelineConfig makePipelineConfig(
    const tt::config::LLMConfig& config) {
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
      return pl::SocketConfig{
          .h2d_socket_id = tt::config::blazeSocketDescriptorPrefix() + "_h2d",
          .d2h_socket_id = tt::config::blazeSocketDescriptorPrefix() + "_d2h",
          .connect_timeout_ms = tt::config::pmConnectTimeoutMs(),
          .use_deepseek_md_format = tt::config::useDeepseekMdFormat()};
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return pl::PipelineSimulatorConfig{
          .num_stages = 64,
          .stage_duration_us = 44,
          .decode_token_id = 12345,
      };
      /* spec decode config
       return PipelineSimulatorConfig{
          .num_stages = 64,
          .stage_duration_us = 44,
          .accept_rate = 0.9f,
          .safe_vocab_base = 1000,    // anything safely above your tokenizer's
      stop ids .safe_vocab_modulus = 64,   // any size >= 5; bigger = lower
      coincidental-stop chance
      };
       */
    default:
      throw std::runtime_error("Invalid blaze runner type");
  }
}

}  // namespace tt::runners::blaze::utils
