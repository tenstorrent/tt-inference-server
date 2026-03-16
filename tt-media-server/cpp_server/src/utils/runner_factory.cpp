// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"
#include "config/settings.hpp"
#include "runners/llm_runner.hpp"
#include "runners/embedding_runner.hpp"
#include "utils/logger.hpp"
#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <iostream>

namespace tt::utils::runner_factory {

std::unique_ptr<runners::IRunner> create_runner(
    config::ModelService service,
    const config::RunnerConfig& config,
    ipc::TokenRingBuffer<65536>* result_queue,
    llm_engine::ITaskQueue* task_queue) {

    switch (service) {
        case config::ModelService::EMBEDDING: {
            TT_LOG_INFO("[RunnerFactory] Creating Embedding runner");
            return std::make_unique<runners::EmbeddingRunner>("device_0", 0);
        }
        case config::ModelService::LLM:
        default: {
            TT_LOG_INFO("[RunnerFactory] Creating LLM runner");
            auto& cfg = std::get<config::LLMConfig>(config);

            // Choose runner based on config.runner_type
            if (cfg.runner_type == config::ModelRunnerType::Pipeline) {
                TT_LOG_INFO("[RunnerFactory] Creating SP Pipeline runner");
                return std::make_unique<runners::SpPipelineRunner>(cfg, result_queue, task_queue);
            }

            TT_LOG_INFO("[RunnerFactory] Creating LLM runner");
            return std::make_unique<tt::runners::LLMRunner>(cfg, result_queue, task_queue);
        }
    }
}

} // namespace tt::utils::runner_factory
