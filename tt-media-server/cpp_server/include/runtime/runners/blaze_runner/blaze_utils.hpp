// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "domain/llm/sequence.hpp"
#include "runtime/runners/blaze_runner/blaze_types.hpp"
#include "runtime/runners/blaze_runner/mock_scheduler.hpp"
#include "runtime/runners/blaze_runner/scheduler_interface.hpp"
#include "scheduler/decode/mock_migration_client.hpp"
#ifdef ENABLE_BLAZE_MIGRATION
#include "scheduler/migration_layer_client_adapter.hpp"
#endif
#include "scheduler/mock_migration_client.hpp"
#include "tt_llm_engine/pipeline/channel_configs.hpp"
#include "tt_llm_engine/pipeline/prefill_pipeline_config.hpp"
#include "tt_llm_engine/scheduler/decode/decode_types.hpp"
#include "tt_llm_engine/scheduler/migration_client_interface.hpp"
#include "tt_llm_engine/scheduler/prefill/prefill_types.hpp"
#include "utils/logger.hpp"

namespace {

const char* requestTypeToString(tt_llm_engine::scheduler::RequestType type) {
  switch (type) {
    case tt_llm_engine::scheduler::RequestType::ALLOCATE:
      return "ALLOCATE";
    case tt_llm_engine::scheduler::RequestType::SUBMIT:
      return "SUBMIT";
    case tt_llm_engine::scheduler::RequestType::CONTINUE:
      return "CONTINUE";
    case tt_llm_engine::scheduler::RequestType::EVICT:
      return "EVICT";
    case tt_llm_engine::scheduler::RequestType::STOP:
      return "STOP";
  }
  return "UNKNOWN";
}

template <typename T>
std::string formatOptional(const std::optional<T>& value) {
  return value.has_value() ? std::to_string(*value) : "none";
}

}  // namespace

namespace tt::runners::blaze::utils {

namespace sch = tt_llm_engine::scheduler;
namespace ds = sch::decode;
namespace ps = sch::prefill;

inline void logISRequest(const sch::ISRequest& req) {
  const sch::GenerationParams& gen = req.gen;
  TT_LOG_DEBUG(
      "ISRequest type={} request_id={} slot_id={} token_count={} "
      "position_id={} dest_slot_id={} migration_uuid={} "
      "migration_start_position={} migrate_from_slot={} "
      "gen.max_new_tokens={} gen.spec_decode={} gen.ignore_eos={} "
      "gen.sampling.temp={} gen.sampling.top_p={} gen.sampling.top_k={} "
      "gen.reasoning_sampling.temp={} gen.reasoning_sampling.top_p={} "
      "gen.reasoning_sampling.top_k={} gen.disaggregated_decode={} "
      "gen.starts_in_thinking={} gen.await_kv_migration={} gen.prefill_only={} "
      "gen.relaxed_acceptance_threshold={} gen.stop_token_count={}",
      requestTypeToString(req.type), req.request_id, req.slot_id,
      req.tokens.size(), formatOptional(req.position_id),
      formatOptional(req.dest_slot_id), formatOptional(req.migration_uuid),
      formatOptional(req.migration_start_position),
      formatOptional(req.migrate_from_slot), gen.max_new_tokens,
      gen.spec_decode, gen.ignore_eos, gen.sampling.temperature,
      gen.sampling.top_p, gen.sampling.top_k,
      gen.reasoning_sampling.temperature, gen.reasoning_sampling.top_p,
      gen.reasoning_sampling.top_k, gen.disaggregated_decode,
      gen.starts_in_thinking, gen.await_kv_migration, gen.prefill_only,
      gen.relaxed_acceptance_threshold, gen.stop_tokens.size());
}

inline sch::ISRequest makeAllocateRequest(
    uint32_t requestId,
    std::optional<uint32_t> migrateFromSlot = std::nullopt) {
  auto req = sch::ISRequest{
      .type = ds::RequestType::ALLOCATE,
      .request_id = requestId,
      .tokens = {},
      .gen = {},
  };
  if (tt::config::enableMigration() and migrateFromSlot.has_value()) {
    req.migrate_from_slot = *migrateFromSlot;
  }
  return req;
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
  const sch::PhaseSamplingParams userSampling{
      .temperature = seq.getSamplingParams().temperature,
      .top_p = seq.getSamplingParams().top_p.value_or(1.0f),
      .top_k = static_cast<int32_t>(seq.getSamplingParams().top_k.value_or(-1)),
  };
  return {
      .max_new_tokens =
          static_cast<uint32_t>(seq.getSamplingParams().max_tokens.value_or(
              static_cast<int>(tt::config::maxContextLength()))),
      .spec_decode = seq.getSamplingParams().fast_mode,
      .ignore_eos = seq.getSamplingParams().ignore_eos,
      .sampling = userSampling,
      .reasoning_sampling = userSampling,
      .disaggregated_decode =
          tt::config::enableMigration() && seq.isDisaggregated(),
      .starts_in_thinking = seq.getStartsInThinking(),
      .stop_tokens = seq.getSamplingParams().stop_token_ids,
  };
}

inline void postProcessSamplingParams(sch::GenerationParams& params) {
  if (tt::config::sampleOnlyInReasoning()) {
    // We argmax outside the reasoning phase
    params.sampling = sch::PhaseSamplingParams{
        .temperature = 1.0f, .top_p = 1.0f, .top_k = 1};
  }
}

inline void fillSequenceFields(sch::ISRequest& req,
                               const tt::domain::llm::Sequence& seq) {
  req.tokens.assign(seq.getTokenIds().begin(), seq.getTokenIds().end());
  req.gen = makeGenerationParams(seq);
  if (seq.getKVPositionId().has_value()) {  // override position id
    req.position_id = *seq.getKVPositionId();
  }
  postProcessSamplingParams(req.gen);
  if (tt::config::enableMigration()) {
    req.migration_uuid = seq.getMigrationId();
  }
}

inline sch::ISRequest makeSubmitRequest(
    uint32_t slotId, const tt::domain::llm::Sequence& seq,
    std::optional<uint32_t> destSlotId = std::nullopt) {
  sch::ISRequest req{};
  req.type = ds::RequestType::SUBMIT;
  req.slot_id = slotId;
  if (tt::config::enableMigration()) {
    req.migration_start_position = seq.getMigrationStartPosition();
    req.dest_slot_id = destSlotId;
  }
  fillSequenceFields(req, seq);
  logISRequest(req);
  return req;
}

inline sch::ISRequest makeContinueRequest(
    uint32_t slotId, const tt::domain::llm::Sequence& seq,
    std::optional<uint32_t> destSlotId = std::nullopt) {
  sch::ISRequest req{};
  req.type = ds::RequestType::CONTINUE;
  req.slot_id = slotId;
  if (tt::config::enableMigration()) {
    req.dest_slot_id = destSlotId;
  }
  fillSequenceFields(req, seq);
  return req;
}

// Populates per-run fields on `slot` from `seq`. Snapshots the slot's spec
// counters at this moment so handleOutput can later report per-turn deltas.
// Does not touch state machine / metrics / task binding - caller's job.
inline void initSlotForRun(SlotContext& slot,
                           const tt::domain::llm::Sequence& seq,
                           IDecodeScheduler& sched) {
  slot.ignoreEos = seq.getSamplingParams().ignore_eos;
  slot.specAcceptsAtStart = sched.get_spec_accepts(slot.slotId);
  slot.specRejectsAtStart = sched.get_spec_rejects(slot.slotId);
  slot.taskId = seq.taskId;
  slot.tokensGenerated = 0;
}

// Prefill overload: no spec-decode counters to snapshot.
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
inline SpecDelta computeAndLogSpecDelta(IDecodeScheduler& sched,
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
      return pl::PrefillMockConfig{
          .auto_layer_acks = true,
          .chunk_latency = std::chrono::milliseconds(100)};
    default:
      throw std::runtime_error("Invalid blaze prefill runner type");
  }
}

inline pl::CounterChannelConfig makePrefillAckChannelConfig(
    const tt::config::LLMConfig& config) {
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
      return pl::InterProcessCounterChannelConfig{
          .shm_name = tt::config::prefillAckChannelName(),
          .connect_timeout_ms = tt::config::pmConnectTimeoutMs(),
      };
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return pl::SingleProcessCounterChannelConfig{};
    default:
      throw std::runtime_error("Invalid blaze prefill runner type");
  }
}

// Builders for the mock scheduler config structs. Same shape as the pipeline
// builders above (env/settings -> plain-data config): the callers in
// blaze_scheduler_factory.cpp have already branched on MOCK_PIPELINE +
// useMockScheduler, so these deliberately take no arguments.
inline MockPrefillSchedulerConfig makeMockPrefillSchedulerConfig() {
  return MockPrefillSchedulerConfig{
      .prefillLatency =
          std::chrono::milliseconds(tt::config::mockPrefillLatencyMs()),
      .prefillChunkSize = tt::config::prefillChunkSize(),
  };
}

inline MockDecodeSchedulerConfig makeMockDecodeSchedulerConfig() {
  return MockDecodeSchedulerConfig{
      .prefillLatency =
          std::chrono::milliseconds(tt::config::mockPrefillLatencyMs()),
      .prefillChunkSize = tt::config::prefillChunkSize(),
      .decodeTokenId = tt::config::mockDecodeTokenId(),
      .decodeTokenLatency =
          std::chrono::microseconds(tt::config::mockDecodeTokenLatencyUs()),
  };
}

inline std::unique_ptr<sch::MigrationClientInterface>
makeMigrationClientInterface(const tt::config::LLMConfig& config) {
  if (!tt::config::enableMigration()) {
    return nullptr;
  }
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
#ifdef ENABLE_BLAZE_MIGRATION
      // create() waits (retries) for the migration endpoint's shmem queues to appear
      // instead of throwing immediately, so the server can start before the endpoint
      // is up. Timeout via TT_MIGRATION_CLIENT_ATTACH_TIMEOUT_MS (default 2 min).
      return sch::MigrationLayerClientAdapter::create(
          tt::config::migrationCmdQueueName(),
          tt::config::migrationTableQueueName(),
          tt::config::migrationRespQueueName());
#else
      throw std::runtime_error(
          "LLM_DEVICE_BACKEND=pipeline_manager requires a build with "
          "--blaze-with-migration");
#endif
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return std::make_unique<sch::MockMigrationClient>();
    default:
      throw std::runtime_error("Invalid blaze decode runner type");
  }
}

inline std::unique_ptr<sch::MigrationClientInterface>
makeDecodeMigrationClientInterface(const tt::config::LLMConfig& config) {
  if (!tt::config::enableMigration()) {
    return nullptr;
  }
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
#ifdef ENABLE_BLAZE_MIGRATION
      // create() waits (retries) for the migration endpoint's shmem queues to appear
      // instead of throwing immediately, so the server can start before the endpoint
      // is up. Timeout via TT_MIGRATION_CLIENT_ATTACH_TIMEOUT_MS (default 2 min).
      return sch::MigrationLayerClientAdapter::create(
          tt::config::migrationCmdQueueName(),
          tt::config::migrationTableQueueName(),
          tt::config::migrationRespQueueName());
#else
      throw std::runtime_error(
          "LLM_DEVICE_BACKEND=pipeline_manager requires a build with "
          "--blaze-with-migration");
#endif
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return std::make_unique<ds::MockMigrationClient>();
    default:
      throw std::runtime_error("Invalid blaze decode runner type");
  }
}

}  // namespace tt::runners::blaze::utils
