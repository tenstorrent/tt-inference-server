// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"
#include "config/settings.hpp"
#include "runners/llm_runner.hpp"
#include "runners/embedding_runner.hpp"
#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <iostream>

namespace tt::utils::runner_factory {

std::unique_ptr<runners::IRunner> create_runner(
    config::ModelService service,
    const runners::RunnerConfig& config,
    ipc::TokenRingBuffer<65536>* result_queue,
    llm_engine::ITaskQueue* task_queue) {

    switch (service) {
        case config::ModelService::EMBEDDING: {
            std::cout << "[RunnerFactory] Creating Embedding runner\n" << std::flush;
            return std::make_unique<runners::EmbeddingRunner>("device_0", 0);
        }
        case config::ModelService::LLM:
        default: {
            if (auto* sp_cfg = std::get_if<sp_pipeline::SpPipelineConfig>(&config)) {
                std::cout << "[RunnerFactory] Creating SP Pipeline runner\n" << std::flush;
                return std::make_unique<runners::SpPipelineRunner>(*sp_cfg, result_queue, task_queue);
            }
            std::cout << "[RunnerFactory] Creating LLM runner\n" << std::flush;
            auto& cfg = std::get<llm_engine::Config>(config);
            return std::make_unique<tt::runners::LLMRunner>(cfg, result_queue, task_queue);
        }
    }
}

} // namespace tt::utils::runner_factory
