// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"
#include "config/settings.hpp"
#include "runners/llm_runner.hpp"
#include "runners/embedding_runner.hpp"

#include <iostream>
#include <stdexcept>

namespace tt::utils::runner_factory {

std::unique_ptr<runners::IRunner> create_runner(
    const runners::RunnerConfig& config,
    llm_engine::TokenCallback on_token,
    llm_engine::ITaskQueue* task_queue) {

    return std::visit([&](const auto& cfg) -> std::unique_ptr<runners::IRunner> {
        using T = std::decay_t<decltype(cfg)>;

        if constexpr (std::is_same_v<T, llm_engine::Config>) {
            std::cout << "[RunnerFactory] Creating LLM runner\n" << std::flush;
            return std::make_unique<tt::runners::LLMRunner>(cfg, std::move(on_token), task_queue);
        } else if constexpr (std::is_same_v<T, runners::EmbeddingConfig>) {
            std::cout << "[RunnerFactory] Creating Embedding runner\n" << std::flush;
            return std::make_unique<runners::EmbeddingRunner>(cfg.device_id, cfg.visible_device);
        } else {
            std::cout << "[RunnerFactory] Unknown runner config type\n, defaulting to LLM runner" << std::flush;
            return std::make_unique<tt::runners::LLMRunner>(cfg, std::move(on_token), task_queue);
        }
    }, config);
}

} // namespace tt::utils::runner_factory
