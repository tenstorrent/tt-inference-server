// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"
#include "config/settings.hpp"
#include "runners/llm_runner.hpp"
#include "runners/embedding_runner.hpp"

#include <iostream>

namespace tt::utils::runner_factory {

std::unique_ptr<runners::IRunner> create_runner(
    const llm_engine::Config& config,
    llm_engine::TokenCallback on_token,
    llm_engine::ITaskQueue* task_queue) {
    
    std::string runner_type = tt::config::runner_type();
    
    if (runner_type == "llm") {
        std::cout << "[RunnerFactory] Creating LLM runner\n" << std::flush;
        return std::make_unique<llm_engine::LLMRunner>(config, std::move(on_token), task_queue);
    } else if (runner_type == "embedding") {
        std::cout << "[RunnerFactory] Creating Embedding runner\n" << std::flush;
        // For embedding runner, we'll use device_0 as default and visible_device=0
        return std::make_unique<runners::EmbeddingRunner>("device_0", 0);
    } else {
        std::cout << "[RunnerFactory] Unknown runner type '" << runner_type 
                  << "', defaulting to LLM runner\n" << std::flush;
        return std::make_unique<llm_engine::LLMRunner>(config, std::move(on_token), task_queue);
    }
}

} // namespace tt::utils::runner_factory
