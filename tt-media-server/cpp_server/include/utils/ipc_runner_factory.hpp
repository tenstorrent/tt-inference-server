// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "ipc/interface/cancel_queue.hpp"
#include "ipc/interface/result_queue.hpp"
#include "ipc/interface/task_queue.hpp"
#include "runtime/runners/ipc_runner.hpp"

namespace tt::utils::ipc_runner_factory {

/** Worker-process entry point for IPC runners (LLM, embedding). Delegates
 *  to `RunnerRegistry::createIpc`; media runners go directly through
 *  `service_factory` instead. */
std::unique_ptr<runners::IRunner> createIpcRunner(
    config::ModelService service, const config::RunnerConfig& config,
    ipc::IResultQueue* resultQueue, tt::ipc::ITaskQueue* taskQueue,
    ipc::ICancelQueue* cancelQueue = nullptr);

}  // namespace tt::utils::ipc_runner_factory
