// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "config/runner_config.hpp"
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
#include "utils/logger.hpp"
#ifdef KAFKA_ENABLED
#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "services/composite_migration_client.hpp"
#include "services/remote_kv_manager_adapter.hpp"
#include "services/remote_kv_manager_impl.hpp"
#endif

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

inline void logISRequest(const sch::ISRequest& req) {
  const sch::GenerationParams& gen = req.gen;
  TT_LOG_DEBUG(
      "ISRequest type={} request_id={} slot_id={} token_count={} "
      "position_id={} dest_slot_id={} migration_uuid={} "
      "migration_start_position={} migrate_from_slot={} "
      "gen.max_new_tokens={} gen.spec_decode={} gen.ignore_eos={} "
      "gen.sampling.temp={} gen.sampling.top_p={} gen.sampling.top_k={} "
      "gen.disaggregated_decode={} "
      "gen.starts_in_thinking={} gen.await_kv_migration={} gen.prefill_only={} "
      "gen.relaxed_acceptance_threshold={} gen.stop_token_count={}",
      requestTypeToString(req.type), req.request_id, req.slot_id,
      req.tokens.size(), formatOptional(req.position_id),
      formatOptional(req.dest_slot_id), formatOptional(req.migration_uuid),
      formatOptional(req.migration_start_position),
      formatOptional(req.migrate_from_slot), gen.max_new_tokens,
      gen.spec_decode, gen.ignore_eos, gen.sampling.temperature,
      gen.sampling.top_p, gen.sampling.top_k, gen.disaggregated_decode,
      gen.starts_in_thinking, gen.await_kv_migration, gen.prefill_only,
      gen.relaxed_acceptance_threshold, gen.stop_tokens.size());
}

inline sch::ISRequest makeAllocateRequest(
    const tt::config::BlazeConfig& config, uint32_t requestId,
    std::optional<uint32_t> migrateFromSlot = std::nullopt) {
  auto req = sch::ISRequest{
      .type = ds::RequestType::ALLOCATE,
      .request_id = requestId,
      .tokens = {},
      .gen = {},
  };
  if (config.enableMigration and migrateFromSlot.has_value()) {
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
    const tt::config::BlazeConfig& config,
    const tt::domain::llm::Sequence& seq) {
  const sch::SamplingParams sampling{
      .temperature = seq.getSamplingParams().temperature,
      .top_p = seq.getSamplingParams().top_p.value_or(1.0f),
      .top_k = static_cast<int32_t>(seq.getSamplingParams().top_k.value_or(-1)),
  };
  return {
      .max_new_tokens =
          static_cast<uint32_t>(seq.getSamplingParams().max_tokens.value_or(
              static_cast<int>(config.maxContextLength))),
      .spec_decode = seq.getSamplingParams().fast_mode,
      .ignore_eos = seq.getSamplingParams().ignore_eos,
      .sampling = sampling,
      .disaggregated_decode = config.enableMigration && seq.isDisaggregated(),
      .starts_in_thinking = seq.getStartsInThinking(),
      .stop_tokens = seq.getSamplingParams().stop_token_ids,
  };
}

inline void fillSequenceFields(const tt::config::BlazeConfig& config,
                               sch::ISRequest& req,
                               const tt::domain::llm::Sequence& seq) {
  req.tokens.assign(seq.getTokenIds().begin(), seq.getTokenIds().end());
  req.gen = makeGenerationParams(config, seq);
  if (seq.getKVPositionId().has_value()) {  // override position id
    req.position_id = *seq.getKVPositionId();
  }
  if (config.enableMigration) {
    req.migration_uuid = seq.getMigrationId();
  }
}

inline sch::ISRequest makeSubmitRequest(
    const tt::config::BlazeConfig& config, uint32_t slotId,
    const tt::domain::llm::Sequence& seq,
    std::optional<uint32_t> destSlotId = std::nullopt) {
  sch::ISRequest req{};
  req.type = ds::RequestType::SUBMIT;
  req.slot_id = slotId;
  if (config.enableMigration) {
    req.migration_start_position = seq.getMigrationStartPosition();
    req.dest_slot_id = destSlotId;
  }
  fillSequenceFields(config, req, seq);
  logISRequest(req);
  return req;
}

inline sch::ISRequest makeContinueRequest(
    const tt::config::BlazeConfig& config, uint32_t slotId,
    const tt::domain::llm::Sequence& seq,
    std::optional<uint32_t> destSlotId = std::nullopt) {
  sch::ISRequest req{};
  req.type = ds::RequestType::CONTINUE;
  req.slot_id = slotId;
  if (config.enableMigration) {
    req.dest_slot_id = destSlotId;
  }
  fillSequenceFields(config, req, seq);
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
    const tt::config::BlazeConfig& config) {
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
      return pl::SocketConfig{
          .h2d_socket_id = config.blazeSocketDescriptorPrefix,
          .d2h_socket_id = config.blazeSocketDescriptorPrefix,
          .connect_timeout_ms = config.pmConnectTimeoutMs,
          .wire_format = wireFormatFromString(config.wireFormat)};
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return pl::PipelineSimulatorConfig{
          .num_stages = config.numPipelineStages,
          .stage_duration_us = config.mockStageLatencyUs,
          .decode_token_id = config.mockDecodeTokenId,
      };
    default:
      throw std::runtime_error("Invalid blaze decode runner type");
  }
}

inline pl::PrefillPipelineConfig makePrefillPipelineConfig(
    const tt::config::BlazeConfig& config) {
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
      return pl::PrefillH2DConfig{
          .service_id = config.blazeSocketDescriptorPrefix,
          .connect_timeout_ms = config.pmConnectTimeoutMs};
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return pl::PrefillMockConfig{
          .auto_layer_acks = true,
          .chunk_latency = std::chrono::milliseconds(100)};
    default:
      throw std::runtime_error("Invalid blaze prefill runner type");
  }
}

inline pl::CounterChannelConfig makePrefillAckChannelConfig(
    const tt::config::BlazeConfig& config) {
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
      return pl::InterProcessCounterChannelConfig{
          .shm_name = config.prefillAckChannelName,
          .connect_timeout_ms = config.pmConnectTimeoutMs,
      };
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return pl::SingleProcessCounterChannelConfig{};
    default:
      throw std::runtime_error("Invalid blaze prefill runner type");
  }
}

// Builders for the mock scheduler config structs. Same shape as the pipeline
// builders above (config -> plain-data config): the callers in
// blaze_scheduler_factory.cpp have already branched on MOCK_SCHEDULER, so
// these read only the mock knobs off `config`.
inline MockPrefillSchedulerConfig makeMockPrefillSchedulerConfig(
    const tt::config::BlazeConfig& config) {
  return MockPrefillSchedulerConfig{
      .prefillLatency = std::chrono::milliseconds(config.mockPrefillLatencyMs),
      .prefillChunkSize = config.prefillChunkSize,
  };
}

inline MockDecodeSchedulerConfig makeMockDecodeSchedulerConfig(
    const tt::config::BlazeConfig& config) {
  return MockDecodeSchedulerConfig{
      .numPipelineStages = config.numPipelineStages,
      .stageLatency = std::chrono::microseconds(config.mockStageLatencyUs),
      .decodeTokenId = config.mockDecodeTokenId,
  };
}

// Runner-type-driven migration client factory that hands out the shmem-backed
// MigrationLayerClientAdapter for real device runs, or a MockMigrationClient
// for the mock pipeline. This is the client the PrefillScheduler gets when
// PREFILL_USE_REMOTE_KV_MANAGER is off, AND — when it is on — the loopback
// backend that a CompositeMigrationClient uses to serve migrate() calls the
// Kafka RemoteKVManagerAdapter cannot handle (ALLOCATE(migrate_from_slot)
// prefix-cache slot copies). Kept as a separate helper so both callers share
// exactly one runner_type switch.
//
// NOTE: this is not the top-level factory called by the scheduler factory —
// that is makeMigrationClientInterface() below, which dispatches to either
// this helper or the Kafka path.
inline std::unique_ptr<sch::MigrationClientInterface>
makeShmemOrMockMigrationClient(const tt::config::BlazeConfig& config) {
  if (!config.enableMigration) {
    return nullptr;
  }
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
#ifdef ENABLE_BLAZE_MIGRATION
      return std::make_unique<sch::MigrationLayerClientAdapter>(
          config.migrationCmdQueueName, config.migrationTableQueueName,
          config.migrationRespQueueName);
#else
      throw std::runtime_error(
          "LLM_DEVICE_BACKEND=pipeline_manager requires a build with "
          "--blaze-with-migration");
#endif
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return std::make_unique<sch::MockMigrationClient>();
    default:
      throw std::runtime_error("Invalid blaze prefill runner type");
  }
}

// Kafka-backed migration client for the PrefillScheduler. Builds two
// backends and hands them to a CompositeMigrationClient:
//
//   burst backend (Kafka):
//     RemoteKVManagerAdapter wraps a RemoteKVManagerImpl wired to a Kafka
//     producer (request topic) and Kafka consumer (ack topic). Handles the
//     start_burst / enqueue_migration_in_burst / finish_burst path used for
//     cross-endpoint P->D KV migration on a migrating SUBMIT.
//
//   loopback backend (shmem or mock, per runner_type):
//     Whatever makeShmemOrMockMigrationClient would return for the current
//     runner_type. Serves the single-shot migrate() path used for
//     ALLOCATE(migrate_from_slot) prefix-cache slot copies, which the Kafka
//     adapter intentionally does not implement (RemoteKVManagerAdapter::
//     migrate() throws). Without this, a single IS request with
//     migrate_from_slot would crash the prefill worker.
//
// Only compiled when the binary was built with KAFKA_ENABLED=ON; the
// non-Kafka build surfaces a clear error at scheduler construction (in
// makeMigrationClientInterface below) instead of silently downgrading to
// the shmem-only path.
#ifdef KAFKA_ENABLED
inline std::unique_ptr<sch::MigrationClientInterface>
makePrefillKafkaMigrationClient(const tt::config::BlazeConfig& config) {
  auto requestProducer = std::make_unique<tt::messaging::KafkaProducer>(
      tt::messaging::KafkaProducerConfig{
          .brokers = tt::config::kafkaBrokers(),
          .topic = tt::config::kafkaMigrationRequestTopic(),
      });
  auto ackConsumer = std::make_unique<tt::messaging::KafkaConsumer>(
      tt::messaging::KafkaConsumerConfig{
          .brokers = tt::config::kafkaBrokers(),
          .topic = tt::config::kafkaMigrationAckTopic(),
          // Shared with the worker's request-topic subscription: group.id is
          // scoped per topic, so worker (request topic) and client (ack topic)
          // do not fight over partitions. If multiple PrefillScheduler
          // processes need to consume acks independently, they must set
          // KAFKA_GROUP_ID to distinct values at deploy time — sharing a
          // group here would split ack partitions across processes and
          // silently starve some bursts of their completion event.
          .group_id = tt::config::kafkaGroupId(),
      });
  auto kvManager = std::make_unique<tt::services::RemoteKVManagerImpl>(
      std::move(requestProducer), std::move(ackConsumer),
      std::chrono::milliseconds(tt::config::kvMigrationTimeoutMs()),
      std::chrono::milliseconds(tt::config::kvMigrationSweepIntervalMs()),
      static_cast<int>(tt::config::kvMigrationDrainPollMs()));
  auto burst = std::make_unique<tt::services::RemoteKVManagerAdapter>(
      std::move(kvManager));
  auto loopback = makeShmemOrMockMigrationClient(config);
  TT_LOG_INFO(
      "makePrefillKafkaMigrationClient: CompositeMigrationClient wired "
      "(brokers={}, request_topic={}, ack_topic={}, group_id={}); burst = "
      "RemoteKVManagerAdapter, loopback per runner_type",
      tt::config::kafkaBrokers(), tt::config::kafkaMigrationRequestTopic(),
      tt::config::kafkaMigrationAckTopic(), tt::config::kafkaGroupId());
  return std::make_unique<tt::services::CompositeMigrationClient>(
      std::move(burst), std::move(loopback));
}
#endif  // KAFKA_ENABLED

// Top-level migration-client factory for the PrefillScheduler. Dispatches
// on the (env-derived) prefillUseRemoteKvManager flag carried on the config:
// when true, the burst path goes through the Kafka-backed
// RemoteKVManagerAdapter (composed with a shmem/mock loopback for migrate()
// calls the adapter cannot service); when false, the scheduler talks directly
// to the shmem/mock client. This function is what
// blaze_scheduler_factory::makePrefillScheduler calls; the two helpers above
// are private-in-spirit implementation details.
inline std::unique_ptr<sch::MigrationClientInterface>
makeMigrationClientInterface(const tt::config::BlazeConfig& config) {
  if (!config.enableMigration) {
    return nullptr;
  }
  if (config.prefillUseRemoteKvManager) {
#ifdef KAFKA_ENABLED
    return makePrefillKafkaMigrationClient(config);
#else
    throw std::runtime_error(
        "PREFILL_USE_REMOTE_KV_MANAGER=1 but this binary was built without "
        "KAFKA_ENABLED=ON. Rebuild with -DKAFKA_ENABLED=ON or unset "
        "PREFILL_USE_REMOTE_KV_MANAGER to use the shmem MigrationLayerClient.");
#endif
  }
  return makeShmemOrMockMigrationClient(config);
}

inline std::unique_ptr<sch::MigrationClientInterface>
makeDecodeMigrationClientInterface(const tt::config::BlazeConfig& config) {
  if (!config.enableMigration) {
    return nullptr;
  }
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
#ifdef ENABLE_BLAZE_MIGRATION
      return std::make_unique<sch::MigrationLayerClientAdapter>(
          config.migrationCmdQueueName, config.migrationTableQueueName,
          config.migrationRespQueueName);
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
