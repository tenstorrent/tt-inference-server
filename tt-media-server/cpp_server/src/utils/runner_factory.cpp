// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"
#include "config/settings.hpp"
#include "runners/llm_runner.hpp"
#include "runners/embedding_runner.hpp"

#include <iostream>

namespace tt::utils::runner_factory {

std::unique_ptr<runners::IRunner> create_runner(
    config::ModelService service,
    const runners::RunnerConfig& config,
    llm_engine::TokenCallback on_token,
    llm_engine::ITaskQueue* task_queue) {

    switch (service) {
        case config::ModelService::EMBEDDING: {
            std::cout << "[RunnerFactory] Creating Embedding runner\n" << std::flush;
            return std::make_unique<runners::EmbeddingRunner>("device_0", 0);
        }
        case config::ModelService::LLM:
        default: {
            std::cout << "[RunnerFactory] Creating LLM runner\n" << std::flush;
            auto& cfg = std::get<llm_engine::Config>(config);
            return std::make_unique<tt::runners::LLMRunner>(cfg, std::move(on_token), task_queue);
        }
    }
}

} // namespace tt::utils::runner_factory
