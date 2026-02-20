// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>

#include "runners/runner_interface.hpp"
#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/task_queue.hpp"

namespace llm_engine {
    using TokenCallback = std::function<void(TaskID task_id, uint64_t token_id, bool finished)>;
}

namespace tt::utils::runner_factory {

/**
 * Create a runner based on the current configuration.
 * The runner type is determined by the MODEL_SERVICE environment variable.
 * 
 * @param config LLM engine configuration
 * @param on_token Token callback for streaming
 * @param task_queue Task queue for worker communication
 * @return Unique pointer to the created runner
 */
std::unique_ptr<runners::IRunner> create_runner(
    const llm_engine::Config& config,
    llm_engine::TokenCallback on_token,
    llm_engine::ITaskQueue* task_queue
);

/**
 * Get the runner type string for the current configuration.
 * @return Runner type string (e.g., "llm", "embedding")
 */
std::string get_runner_type();

} // namespace tt::utils::runner_factory
