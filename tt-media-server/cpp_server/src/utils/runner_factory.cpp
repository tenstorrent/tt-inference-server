// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"

#include "runners/embedding_runner.hpp"
#include "runners/llm_runner.hpp"
#include "runners/sp_pipeline_runner/mock_sp_pipeline_model_runner.hpp"
#include "runners/sp_pipeline_runner/sp_pipeline_model_runner.hpp"
#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"
#include "utils/logger.hpp"

namespace tt::utils::runner_factory {

std::unique_ptr<runners::IRunner> createRunner(
    config::ModelService service, const config::RunnerConfig& config,
    ipc::TokenRingBuffer<65536>* resultQueue,
    llm_engine::ITaskQueue* taskQueue) {
  switch (service) {
    case config::ModelService::EMBEDDING: {
      TT_LOG_INFO("[RunnerFactory] Creating Embedding runner");
      return std::make_unique<runners::EmbeddingRunner>("device_0", 0);
    }
    case config::ModelService::LLM:
    default: {
      auto& cfg = std::get<config::LLMConfig>(config);

      if (cfg.runner_type == config::ModelRunnerType::PIPELINE) {
        TT_LOG_INFO(
            "[RunnerFactory] Creating SP Pipeline runner (shared memory)");
        auto factory = [](sp_pipeline::DecodeCallback cb)
            -> std::unique_ptr<sp_pipeline::ISpPipelineModelRunner> {
          return std::make_unique<sp_pipeline::SpPipelineModelRunner>(
              std::move(cb));
        };
        return std::make_unique<runners::SpPipelineRunner>(cfg, resultQueue,
                                                           taskQueue, factory);
      }

      if (cfg.runner_type == config::ModelRunnerType::MOCK_PIPELINE) {
        TT_LOG_INFO(
            "[RunnerFactory] Creating SP Pipeline runner (mock device)");
        auto factory = [](sp_pipeline::DecodeCallback cb)
            -> std::unique_ptr<sp_pipeline::ISpPipelineModelRunner> {
          return std::make_unique<sp_pipeline::MockSpPipelineModelRunner>(
              std::move(cb));
        };
        return std::make_unique<runners::SpPipelineRunner>(cfg, resultQueue,
                                                           taskQueue, factory);
      }

      TT_LOG_INFO("[RunnerFactory] Creating LLM runner (mock)");
      return std::make_unique<tt::runners::LLMRunner>(cfg, resultQueue,
                                                      taskQueue);
    }
  }
}

}  // namespace tt::utils::runner_factory
