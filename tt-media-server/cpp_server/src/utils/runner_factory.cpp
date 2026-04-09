// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"

#include "runners/embedding_runner.hpp"
#include "runners/llm_runner.hpp"
#ifdef ENABLE_BLAZE
#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"
#include "runners/sp_prefill_runner/sp_prefill_runner.hpp"
#include "sp_pipeline_runner/sp_pipeline_runner_demo.hpp"
#endif
#include "utils/logger.hpp"

namespace tt::utils::runner_factory {

std::unique_ptr<runners::IRunner> createRunner(
    config::ModelService service, const config::RunnerConfig& config,
    ipc::TokenRingBuffer<65536>* resultQueue, llm_engine::ITaskQueue* taskQueue,
    ipc::ICancelQueue* cancelQueue) {
  switch (service) {
    case config::ModelService::EMBEDDING: {
      TT_LOG_INFO("[RunnerFactory] Creating Embedding runner");
      return std::make_unique<runners::EmbeddingRunner>("device_0", 0);
    }
    case config::ModelService::LLM:
    default: {
      auto& cfg = std::get<config::LLMConfig>(config);

#ifdef ENABLE_BLAZE
      if (cfg.runner_type == config::ModelRunnerType::PIPELINE ||
          cfg.runner_type == config::ModelRunnerType::MOCK_PIPELINE) {
        TT_LOG_INFO("[RunnerFactory] Creating SP Pipeline runner");
        return std::make_unique<runners::SpPipelineRunnerDemo>(cfg, resultQueue,
                                                               taskQueue);
      } else if (cfg.runner_type == config::ModelRunnerType::PIPELINE_MANAGER) {
        TT_LOG_INFO("[RunnerFactory] Creating SP Pipeline runner");
        return std::make_unique<runners::SpPipelineRunner>(cfg, resultQueue,
                                                           taskQueue);
      }

      if (cfg.runner_type == config::ModelRunnerType::PREFILL) {
        TT_LOG_INFO("[RunnerFactory] Creating SP Prefill runner");
        return std::make_unique<runners::SpPrefillRunner>(cfg, resultQueue,
                                                          taskQueue);
      }
#endif

      TT_LOG_INFO("[RunnerFactory] Creating LLM runner (mock)");
      return std::make_unique<tt::runners::LLMRunner>(cfg, resultQueue,
                                                      taskQueue, cancelQueue);
    }
  }
}

}  // namespace tt::utils::runner_factory
