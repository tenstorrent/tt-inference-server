// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"
#include "config/settings.hpp"
#include "runners/llm_runner.hpp"
#include "runners/embedding_runner.hpp"
#include "runners/pybind_llama_model_runner.hpp"

#include <iostream>

namespace tt::utils::runner_factory {

static tt::runners::ModelRunnerFactory make_model_runner_factory() {
    if (tt::config::model_runner_type() != tt::config::RunnerType::LLAMA_RUNNER) {
        return nullptr;
    }
    return [](const llm_engine::Config& cfg, llm_engine::DecodeCallback cb) {
        auto runner = llm_engine::make_pybind_llama_model_runner(cfg, cb);
        if (runner) return runner;
        std::cerr << "[RunnerFactory] Pybind Llama runner init failed, using stub\n";
        return llm_engine::make_model_runner(cfg, std::move(cb));
    };
}

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
            return std::make_unique<tt::runners::LLMRunner>(
                cfg, std::move(on_token), task_queue, make_model_runner_factory());
        }
    }
}

} // namespace tt::utils::runner_factory
