// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "ipc/cancel_queue.hpp"
#include "ipc/result_queue.hpp"
#include "ipc/task_queue.hpp"
#include "runners/ipc_runner.hpp"

namespace tt::utils::ipc_runner_factory {

/**
 * Worker-process entry point for IPC runners (LLM, embedding). Ensures
 * `services::registerBuiltinModelServices()` has run, then delegates to
 * `RunnerRegistry::createIpc`. Direct-call media runners don't go through
 * here; they're constructed inline at registration time.
 */
std::unique_ptr<runners::IRunner> createIpcRunner(
    config::ModelService service, const config::RunnerConfig& config,
    ipc::IResultQueue* resultQueue, tt::ipc::ITaskQueue* taskQueue,
    ipc::ICancelQueue* cancelQueue = nullptr);

}  // namespace tt::utils::ipc_runner_factory
