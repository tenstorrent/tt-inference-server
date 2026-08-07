// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "ipc/interface/cancel_queue.hpp"
#include "ipc/interface/result_queue.hpp"
#include "ipc/interface/task_queue.hpp"
#include "ipc/tts_ipc.hpp"
#include "runtime/runners/ipc_runner.hpp"

namespace tt::utils::ipc_runner_factory {

/** Worker-process entry point for IPC runners (LLM, embedding). Delegates
 *  to `RunnerRegistry::createIpc`; media runners go directly through
 *  `service_factory` instead. */
std::unique_ptr<runners::IRunner> createIpcRunner(
    config::ModelService service, const config::RunnerConfig& config,
    ipc::IResultQueue* resultQueue, tt::ipc::ITaskQueue* taskQueue,
    ipc::ICancelQueue* cancelQueue = nullptr);

/** Worker-process entry point for TTS IPC runners. */
std::unique_ptr<runners::IRunner> createTtsIpcRunner(
    const config::RunnerConfig& config, ipc::tts::TtsTaskQueue* taskQueue,
    ipc::tts::TtsAudioChunkQueue* audioQueue, ipc::ICancelQueue* cancelQueue);

}  // namespace tt::utils::ipc_runner_factory
