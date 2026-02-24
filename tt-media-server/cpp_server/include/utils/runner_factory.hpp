// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>

#include "config/constants.hpp"
#include "runners/runner_interface.hpp"
#include "runners/runner_config.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"

namespace llm_engine {
    using TokenCallback = std::function<void(const TokenResult& result)>;
}

namespace tt::utils::runner_factory {

/**
 * Create a runner for the given model service.
 *
 * @param service Which service (LLM, EMBEDDING, …) to create a runner for
 * @param config Runner configuration data
 * @param on_token Token callback for streaming (used by LLM runner)
 * @param task_queue Task queue for worker communication (used by LLM runner)
 * @return Unique pointer to the created runner
 */
std::unique_ptr<runners::IRunner> create_runner(
    config::ModelService service,
    const runners::RunnerConfig& config,
    llm_engine::TokenCallback on_token,
    llm_engine::ITaskQueue* task_queue
);

/**
 * Get the runner type string for the current configuration.
 * @return Runner type string (e.g., "llm", "embedding")
 */
std::string get_runner_type();

} // namespace tt::utils::runner_factory
