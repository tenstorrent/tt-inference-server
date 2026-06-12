// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <string>

#include "config/settings.hpp"
#include "domain/llm/sequence.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/interface/result_queue.hpp"
#include "runtime/runners/blaze_runner/blaze_types.hpp"
#include "tt_llm_engine/pipeline/channel_configs.hpp"
#include "tt_llm_engine/pipeline/prefill_pipeline_config.hpp"
#include "tt_llm_engine/scheduler/decode/decode_scheduler.hpp"
#include "tt_llm_engine/scheduler/decode/decode_types.hpp"
#include "tt_llm_engine/scheduler/prefill/prefill_types.hpp"
#include "utils/logger.hpp"

namespace tt::runners::blaze::utils {

namespace sch = tt_llm_engine::scheduler;
namespace ds = sch::decode;
namespace ps = sch::prefill;

inline sch::ISRequest makeAllocateRequest(uint32_t requestId) {
  return {.type = ds::RequestType::ALLOCATE,
          .request_id = requestId,
          .tokens = {},
          .gen = {},
          .position_id = std::nullopt,
          .dest_slot_id = std::nullopt};
}

inline sch::ISRequest makeEvictRequest(uint32_t requestId, uint32_t slotId) {
  return {.type = ds::RequestType::EVICT,
          .request_id = requestId,
          .slot_id = slotId,
          .tokens = {},
          .gen = {},
          .position_id = std::nullopt,
          .dest_slot_id = std::nullopt};
}

inline sch::ISRequest makeStopRequest(uint32_t requestId, uint32_t slotId) {
  return {.type = ds::RequestType::STOP,
          .request_id = requestId,
          .slot_id = slotId,
          .tokens = {},
          .gen = {},
          .position_id = std::nullopt,
          .dest_slot_id = std::nullopt};
}

inline sch::GenerationParams makeGenerationParams(
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
      .stop_tokens = seq.getSamplingParams().stop_token_ids};
}

inline void fillSequenceFields(sch::ISRequest& req,
                               const tt::domain::llm::Sequence& seq) {
  req.tokens.assign(seq.getTokenIds().begin(), seq.getTokenIds().end());
  req.gen = makeGenerationParams(seq);
}

inline sch::ISRequest makeSubmitRequest(
    uint32_t slotId, const tt::domain::llm::Sequence& seq,
    std::optional<uint32_t> destSlotId = std::nullopt) {
  sch::ISRequest req{};
  req.type = ds::RequestType::SUBMIT;
  req.slot_id = slotId;
  req.dest_slot_id = destSlotId;
  fillSequenceFields(req, seq);
  return req;
}

inline sch::ISRequest makeContinueRequest(
    uint32_t slotId, const tt::domain::llm::Sequence& seq,
    std::optional<uint32_t> destSlotId = std::nullopt) {
  sch::ISRequest req{};
  req.type = ds::RequestType::CONTINUE;
  req.slot_id = slotId;
  req.dest_slot_id = destSlotId;
  fillSequenceFields(req, seq);
  if (seq.getKVPositionId().has_value()) {  // override position id
    req.position_id = *seq.getKVPositionId();
  }
  return req;
}

// ---------------------------------------------------------------------------
// Interface-border logging.
//
// Every line carries the greppable "[BORDER]" tag so a single
// `grep '\[BORDER\]'` surfaces all traffic crossing a runner boundary.
// Direction is encoded in the arrow: ">>" = runner sent to the scheduler,
// "<<" = runner received (from the scheduler, a queue, or dynamo). `role`
// is "prefill" or "decode" so prefill/decode lines can be told apart in a
// shared log. All emit at INFO.
// ---------------------------------------------------------------------------

inline const char* requestTypeName(sch::RequestType t) {
  switch (t) {
    case sch::RequestType::ALLOCATE:
      return "ALLOCATE";
    case sch::RequestType::SUBMIT:
      return "SUBMIT";
    case sch::RequestType::CONTINUE:
      return "CONTINUE";
    case sch::RequestType::EVICT:
      return "EVICT";
    case sch::RequestType::STOP:
      return "STOP";
  }
  return "UNKNOWN";
}

inline std::string optU32(const std::optional<uint32_t>& v) {
  return v.has_value() ? std::to_string(*v) : "-";
}

// Runner -> scheduler. One line per ISRequest the runner pushes.
inline void logSchedTx(const char* role, const sch::ISRequest& req) {
  TT_LOG_INFO(
      "[BORDER] {}>>sched {} req={} slot={} destSlot={} tok={} posId={} "
      "maxNew={} temp={:.3f} topP={:.3f} topK={} ignoreEos={} spec={} "
      "disagg={} stopTok={}",
      role, requestTypeName(req.type), req.request_id, req.slot_id,
      optU32(req.dest_slot_id), req.tokens.size(), optU32(req.position_id),
      req.gen.max_new_tokens, req.gen.temperature, req.gen.top_p, req.gen.top_k,
      req.gen.ignore_eos, req.gen.spec_decode, req.gen.disaggregated_decode,
      req.gen.stop_tokens.size());
}

// Scheduler -> runner. A generated/prefill OutputMessage. `level` lets the
// per-token decode path drop to DEBUG while low-volume callers stay at INFO.
inline void logSchedRxOutput(const char* role, const sch::OutputMessage& out,
                             tt::utils::ZeroOverheadLogger::Level level =
                                 tt::utils::ZeroOverheadLogger::INFO) {
  tt::utils::ZeroOverheadLogger::log(
      level,
      "[BORDER] {}<<sched OUT slot={} tok={} isComplete={} prefillComplete={} "
      "ctxExhausted={} posId={} realPos={} tokensGen={} req={}",
      role, out.slot_id, out.token_id, out.is_complete, out.prefill_complete,
      out.ctx_exhausted, out.position_id, out.real_pos, out.tokens_generated,
      out.request_id);
}

// Runner -> result/response queue. One line per token (or final/abort/error
// sentinel) the runner publishes back toward the client. Pair with the
// matching logSchedRxOutput line (same taskId) to see what the runner forwarded
// vs. what the scheduler produced.
inline void logResultTx(const char* role, uint32_t taskId, uint64_t tokenId,
                        uint32_t flag,
                        tt::utils::ZeroOverheadLogger::Level level =
                            tt::utils::ZeroOverheadLogger::INFO) {
  tt::utils::ZeroOverheadLogger::log(
      level, "[BORDER] {}>>resultq taskId={} tok={} final={} abort={} error={}",
      role, taskId, tokenId,
      static_cast<bool>(flag & tt::ipc::SharedToken::FLAG_FINAL),
      static_cast<bool>(flag & tt::ipc::SharedToken::FLAG_ABORT),
      static_cast<bool>(flag & tt::ipc::SharedToken::FLAG_ERROR));
}

// Memory queue -> runner. A ManageMemoryTask read from the MemoryManager.
inline void logMemQueueRx(const char* role,
                          const tt::domain::ManageMemoryTask& task) {
  TT_LOG_INFO("[BORDER] {}<<memq taskId={} action={} slot={} copyFrom={}", role,
              task.taskId,
              task.action == tt::domain::MemoryManagementAction::ALLOCATE
                  ? "ALLOCATE"
              : task.action == tt::domain::MemoryManagementAction::DEALLOCATE
                  ? "DEALLOCATE"
                  : "MOVE",
              task.slotId, optU32(task.slotIdToCopyFrom));
}

// Task queue -> runner. A Sequence read from the task queue. `slotId` is the
// caller-resolved KV slot (prefill vs decode accessor differs).
inline void logTaskQueueRx(const char* role,
                           const tt::domain::llm::Sequence& seq,
                           uint32_t slotId) {
  TT_LOG_INFO(
      "[BORDER] {}<<taskq taskId={} slot={} promptTok={} totalTok={} "
      "continuation={} disagg={}",
      role, seq.taskId, slotId, seq.getNumPromptTokens(),
      seq.getTokenIds().size(), seq.isContinuation(), seq.isDisaggregated());
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

// Populates per-run fields on `slot` from `seq`. Snapshots the slot's spec
// counters at this moment so handleOutput can later report per-turn deltas.
// Does not touch state machine / metrics / task binding — caller's job.
inline void initSlotForRun(SlotContext& slot,
                           const tt::domain::llm::Sequence& seq) {
  slot.ignoreEos = seq.getSamplingParams().ignore_eos;
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
                                        uint32_t taskId) {
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
      "is_complete={} ignoreEos={} tokensGenerated={}",
      output.slot_id, d.accepts, total, acceptRate, taskId, output.token_id,
      output.is_complete, slot.ignoreEos, slot.tokensGenerated);
  return d;
}

namespace pl = tt_llm_engine::pipeline;

inline pl::WireFormat wireFormatFromString(const std::string& s) {
  static const std::unordered_map<std::string, pl::WireFormat> formats = {
      {"deepseek", pl::WireFormat::DEEPSEEK},
      {"loopback", pl::WireFormat::LOOPBACK},
      {"blaze", pl::WireFormat::BLAZE}};

  auto it = formats.find(s);
  if (it != formats.end()) {
    return it->second;
  }

  throw std::runtime_error("Invalid wire format: " + s);
}

inline pl::PipelineConfig makeDecodePipelineConfig(
    const tt::config::LLMConfig& config) {
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
      return pl::SocketConfig{
          .h2d_socket_id = tt::config::blazeSocketDescriptorPrefix(),
          .d2h_socket_id = tt::config::blazeSocketDescriptorPrefix(),
          .connect_timeout_ms = tt::config::pmConnectTimeoutMs(),
          .wire_format = wireFormatFromString(tt::config::wireFormat())};
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
      throw std::runtime_error("Invalid blaze decode runner type");
  }
}

inline pl::PrefillPipelineConfig makePrefillPipelineConfig(
    const tt::config::LLMConfig& config) {
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
      return pl::PrefillH2DConfig{
          .service_id = tt::config::blazeSocketDescriptorPrefix(),
          .connect_timeout_ms = tt::config::pmConnectTimeoutMs()};
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return pl::PrefillMockConfig{.auto_layer_acks = true};
    default:
      throw std::runtime_error("Invalid blaze prefill runner type");
  }
}

inline pl::CounterChannelConfig makePrefillAckChannelConfig(
    const tt::config::LLMConfig& config) {
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
      return pl::InterProcessCounterChannelConfig{
          .shm_name = "/tt_prefill_layer_acks_" +
                      tt::config::blazeSocketDescriptorPrefix(),
          .connect_timeout_ms = 60000,
      };
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return pl::SingleProcessCounterChannelConfig{};
    default:
      throw std::runtime_error("Invalid blaze prefill runner type");
  }
}

}  // namespace tt::runners::blaze::utils
