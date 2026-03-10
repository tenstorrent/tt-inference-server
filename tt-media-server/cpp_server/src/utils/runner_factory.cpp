// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/runner_factory.hpp"
#include "config/settings.hpp"
#include "runners/llm_runner.hpp"
#include "runners/embedding_runner.hpp"
#include "utils/logger.hpp"

namespace tt::utils::runner_factory {

std::unique_ptr<runners::IRunner> create_runner(
    config::ModelService service,
    const runners::RunnerConfig& config,
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
            auto& cfg = std::get<llm_engine::Config>(config);
            return std::make_unique<tt::runners::LLMRunner>(cfg, result_queue, task_queue);
        }
    }
}

} // namespace tt::utils::runner_factory
