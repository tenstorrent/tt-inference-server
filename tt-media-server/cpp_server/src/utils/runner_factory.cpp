// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/runner_factory.hpp"

#include "runners/blaze_prefill_runner/blaze_prefill_runner.hpp"
#include "runners/embedding_runner.hpp"
#include "runners/llm_runner.hpp"
#ifdef ENABLE_BLAZE
#include "runners/blaze_runner/blaze_runner.hpp"
#endif
#include "utils/logger.hpp"

namespace tt::utils::runner_factory {

std::unique_ptr<runners::IRunner> createRunner(
    config::ModelService service, const config::RunnerConfig& config,
    ipc::IResultQueue* resultQueue,
    tt::runners::llm_engine::ITaskQueue* taskQueue,
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
        TT_LOG_INFO("[RunnerFactory] Creating Blaze runner");
        return std::make_unique<runners::BlazeRunner>(cfg, resultQueue,
                                                      taskQueue);
      }
#endif
      if (cfg.runner_type == config::ModelRunnerType::PREFILL) {
        TT_LOG_INFO("[RunnerFactory] Creating Blaze Prefill runner");
        return std::make_unique<runners::BlazePrefillRunner>(cfg, resultQueue,
                                                             taskQueue);
      }

      TT_LOG_INFO("[RunnerFactory] Creating LLM runner (mock)");
      return std::make_unique<tt::runners::LLMRunner>(cfg, resultQueue,
                                                      taskQueue, cancelQueue);
    }
  }
}

}  // namespace tt::utils::runner_factory
