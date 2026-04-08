// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>

#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "ipc/cancel_queue.hpp"
#include "ipc/token_ring_buffer.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"

namespace tt::utils::runner_factory {

/**
 * Create a runner for the given model service.
 *
 * @param service Which service (LLM, EMBEDDING, …) to create a runner for
 * @param config Runner configuration data
 * @param result_queue Result queue for the runner to push results into
 * @param task_queue Task queue for worker communication (used by LLM runner)
 * @param cancel_queue Cancel queue for abort signals (nullable)
 * @return Unique pointer to the created runner
 */
std::unique_ptr<runners::IRunner> createRunner(
    config::ModelService service, const config::RunnerConfig& config,
    ipc::TokenRingBuffer<65536>* resultQueue, llm_engine::ITaskQueue* taskQueue,
    ipc::ICancelQueue* cancelQueue = nullptr);

/**
 * Get the runner type string for the current configuration.
 * @return Runner type string (e.g., "llm", "embedding")
 */
std::string getRunnerType();

}  // namespace tt::utils::runner_factory
