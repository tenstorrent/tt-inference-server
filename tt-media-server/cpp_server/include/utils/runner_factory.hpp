// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>

#include "config/constants.hpp"
#include "runners/runner_interface.hpp"
#include "runners/runner_config.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "ipc/shared_memory.hpp"
#include "ipc/cancel_queue.hpp"

namespace tt::utils::runner_factory {

/**
 * Create a runner for the given model service.
 *
 * @param service Which service (LLM, EMBEDDING, …) to create a runner for
 * @param config Runner configuration data
 * @param result_queue Result queue for the runner to push results into
 * @param task_queue Task queue for worker communication (used by LLM runner)
 * @return Unique pointer to the created runner
 */
std::unique_ptr<runners::IRunner> create_runner(
    config::ModelService service,
    const runners::RunnerConfig& config,
    ipc::TokenRingBuffer<65536>* result_queue,
    llm_engine::ITaskQueue* task_queue,
    ipc::CancelQueue* cancel_queue = nullptr
);

/**
 * Get the runner type string for the current configuration.
 * @return Runner type string (e.g., "llm", "embedding")
 */
std::string get_runner_type();

} // namespace tt::utils::runner_factory
