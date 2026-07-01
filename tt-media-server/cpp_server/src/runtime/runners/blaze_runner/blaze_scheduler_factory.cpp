// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_scheduler_factory.hpp"

#include <chrono>
#include <memory>
#include <utility>

#include "config/settings.hpp"
#include "runtime/runners/blaze_runner/blaze_utils.hpp"
#include "runtime/runners/blaze_runner/mock_scheduler.hpp"
#include "tt_llm_engine/scheduler/decode/decode_scheduler.hpp"
#include "tt_llm_engine/scheduler/prefill/prefill_scheduler.hpp"
#include "tt_llm_engine/scheduler/scheduler_types.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::runners::blaze {

namespace {

namespace ds = tt_llm_engine::scheduler::decode;
namespace ps = tt_llm_engine::scheduler::prefill;

class RealDecodeScheduler final : public IDecodeScheduler {
 public:
  explicit RealDecodeScheduler(std::unique_ptr<ds::DecodeScheduler> scheduler)
      : impl(std::move(scheduler)) {}

  void start() override { impl->start(); }
  void stop() override { impl->stop(); }
  bool push_request(const ds::ISRequest& request) override {
    return impl->push_request(request);
  }
  bool try_pop_response(ds::SchedulerResponse& response) override {
    return impl->try_pop_response(response);
  }
  bool try_pop_output(ds::OutputMessage& output) override {
    return impl->try_pop_output(output);
  }
  uint32_t get_spec_accepts(uint32_t slotId) const override {
    return impl->get_spec_accepts(slotId);
  }
  uint32_t get_spec_rejects(uint32_t slotId) const override {
    return impl->get_spec_rejects(slotId);
  }

 private:
  std::unique_ptr<ds::DecodeScheduler> impl;
};

class RealPrefillScheduler final : public IPrefillScheduler {
 public:
  explicit RealPrefillScheduler(std::unique_ptr<ps::PrefillScheduler> scheduler)
      : impl(std::move(scheduler)) {}

  void start() override { impl->start(); }
  void stop() override { impl->stop(); }
  bool push_request(const ps::ISRequest& request) override {
    return impl->push_request(request);
  }
  bool try_pop_response(ps::SchedulerResponse& response) override {
    return impl->try_pop_response(response);
  }
  bool try_pop_output(ps::OutputMessage& output) override {
    return impl->try_pop_output(output);
  }

 private:
  std::unique_ptr<ps::PrefillScheduler> impl;
};

}  // namespace

std::unique_ptr<IDecodeScheduler> makeDecodeScheduler(
    const tt::config::LLMConfig& config) {
  const auto maxUsers = static_cast<uint32_t>(tt::config::pmMaxUsers());
  if (config.runner_type == tt::config::ModelRunnerType::MOCK_PIPELINE &&
      tt::config::useMockScheduler()) {
    TT_LOG_INFO(
        "makeDecodeScheduler: using MockDecodeScheduler (single-threaded)");
    return std::make_unique<MockDecodeScheduler>(
        maxUsers,
        MockDecodeSchedulerConfig{
            .prefillLatency = std::chrono::milliseconds(
                tt::config::mockPrefillLatencyMs()),
            .prefillChunkSize = tt::config::prefillChunkSize(),
            .decodeTokenId = tt::config::mockDecodeTokenId(),
            .decodeTokenLatency = std::chrono::microseconds(
                tt::config::mockDecodeTokenLatencyUs()),
        });
  }

  TT_LOG_INFO(
      "makeDecodeScheduler: Constructing DecodeScheduler with SocketConfig...");
  auto pipelineConfig = utils::makeDecodePipelineConfig(config);
  auto migrationClientInterface =
      utils::makeDecodeMigrationClientInterface(config);
  auto thinkTokenIds = tt::utils::tokenizers::thinkTokenIds();
  auto eosTokenId = tt::utils::tokenizers::staticInfo().eosTokenId;
  ds::SchedulerParams managerParams{};
  managerParams.num_layers = tt::config::modelNumLayers();
  managerParams.eos_token = static_cast<uint32_t>(eosTokenId);
  managerParams.think_open_token_id =
      static_cast<uint32_t>(thinkTokenIds.first);
  managerParams.think_close_token_id =
      static_cast<uint32_t>(thinkTokenIds.second);
  managerParams.max_users = maxUsers;
  managerParams.self_endpoint_id = tt::config::migrationDecodeEndpointId();
  if (tt::config::enableMigration()) {
    migrationClientInterface->connect_to(
        tt::config::migrationPrefillEndpointId(), "CONNECTOR", "ds_pd");
  }
  if (tt::config::specDecodeMode() == "mtp") {
    managerParams.spec_decode_mode = ds::SpecDecodeMode::MTP;
    managerParams.max_spec_tokens =
        static_cast<uint32_t>(tt::config::mtpLevel());
  }
  auto scheduler = std::make_unique<RealDecodeScheduler>(
      std::make_unique<ds::DecodeScheduler>(
          pipelineConfig, managerParams, std::move(migrationClientInterface)));
  TT_LOG_INFO("makeDecodeScheduler: DecodeScheduler constructed");
  return scheduler;
}

std::unique_ptr<IPrefillScheduler> makePrefillScheduler(
    const tt::config::LLMConfig& config) {
  const auto maxUsers = static_cast<uint32_t>(tt::config::pmMaxUsers());
  if (config.runner_type == tt::config::ModelRunnerType::MOCK_PIPELINE &&
      tt::config::useMockScheduler()) {
    TT_LOG_INFO(
        "makePrefillScheduler: using MockPrefillScheduler (single-threaded)");
    return std::make_unique<MockPrefillScheduler>(
        maxUsers,
        MockPrefillSchedulerConfig{
            .prefillLatency = std::chrono::milliseconds(
                tt::config::mockPrefillLatencyMs()),
            .prefillChunkSize = tt::config::prefillChunkSize(),
        });
  }

  TT_LOG_INFO(
      "makePrefillScheduler: Constructing PrefillScheduler with "
      "SocketConfig...");
  auto pipelineConfig = utils::makePrefillPipelineConfig(config);
  ps::SchedulerParams managerParams{};
  managerParams.dest_endpoint_id = tt::config::migrationDecodeEndpointId();
  managerParams.self_endpoint_id = tt::config::migrationPrefillEndpointId();
  managerParams.layers_per_chunk = tt::config::modelNumLayers();
  managerParams.chunk_size = tt::config::prefillChunkSize();
  managerParams.max_users = maxUsers;
  auto ackChannelConfig = utils::makePrefillAckChannelConfig(config);
  auto migrationClientInterface = utils::makeMigrationClientInterface(config);
  if (tt::config::enableMigration()) {
    migrationClientInterface->connect_to(
        tt::config::migrationDecodeEndpointId(), "PUBLISHER", "ds_pd");
  }
  auto scheduler = std::make_unique<RealPrefillScheduler>(
      std::make_unique<ps::PrefillScheduler>(
          pipelineConfig, ackChannelConfig, managerParams,
          std::move(migrationClientInterface)));
  TT_LOG_INFO("makePrefillScheduler: PrefillScheduler constructed");
  return scheduler;
}

}  // namespace tt::runners::blaze
