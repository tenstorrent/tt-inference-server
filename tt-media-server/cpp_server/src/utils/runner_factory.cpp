// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"
#include "config/settings.hpp"
#include "runners/llm_runner.hpp"
#include "runners/embedding_runner.hpp"
#include "runners/pipe_llama_model_runner.hpp"

#include <iostream>

namespace tt::utils::runner_factory {

static tt::runners::ModelRunnerFactory make_model_runner_factory() {
    if (tt::config::model_runner_type() != tt::config::RunnerType::TTNN_TEST) {
        return nullptr;
    }
    return [](const llm_engine::Config& cfg, llm_engine::DecodeCallback cb) {
        auto runner = llm_engine::make_pipe_llama_model_runner(cfg, cb);
        if (runner) return runner;
        std::cerr << "[RunnerFactory] Pipe Llama runner spawn failed, using stub\n";
        return llm_engine::make_model_runner(cfg, std::move(cb));
    };
}

std::unique_ptr<runners::IRunner> create_runner(
    const llm_engine::Config& config,
    llm_engine::TokenCallback on_token,
    llm_engine::ITaskQueue* task_queue) {
    
    std::string runner_type = tt::config::runner_type();
    
    if (runner_type == "llm") {
        std::cout << "[RunnerFactory] Creating LLM runner\n" << std::flush;
        return std::make_unique<tt::runners::LLMRunner>(
            config, std::move(on_token), task_queue, make_model_runner_factory());
    } else if (runner_type == "embedding") {
        std::cout << "[RunnerFactory] Creating Embedding runner\n" << std::flush;
        return std::make_unique<runners::EmbeddingRunner>("device_0", 0);
    } else {
        std::cout << "[RunnerFactory] Unknown runner type '" << runner_type 
                  << "', defaulting to LLM runner\n" << std::flush;
        return std::make_unique<tt::runners::LLMRunner>(
            config, std::move(on_token), task_queue, make_model_runner_factory());
    }
}

} // namespace tt::utils::runner_factory
