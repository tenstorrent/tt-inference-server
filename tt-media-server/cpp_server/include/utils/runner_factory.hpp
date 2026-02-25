// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>

#include "config/constants.hpp"
#include "runners/runner_interface.hpp"
#include "runners/runner_config.hpp"
#include "runners/runner_result.hpp"
#include "runners/llm_runner/task_queue.hpp"

namespace tt::utils::runner_factory {

/**
 * Create a runner for the given model service.
 *
 * @param service Which service (LLM, EMBEDDING, …) to create a runner for
 * @param config Runner configuration data
 * @param on_result Generic result callback (used by LLM runner for tokens, embedding runner for vectors, etc.)
 * @param task_queue Task queue for worker communication (used by LLM runner)
 * @return Unique pointer to the created runner
 */
std::unique_ptr<runners::IRunner> create_runner(
    config::ModelService service,
    const runners::RunnerConfig& config,
    runners::ResultCallback on_result,
    llm_engine::ITaskQueue* task_queue
);

/**
 * Get the runner type string for the current configuration.
 * @return Runner type string (e.g., "llm", "embedding")
 */
std::string get_runner_type();

} // namespace tt::utils::runner_factory
