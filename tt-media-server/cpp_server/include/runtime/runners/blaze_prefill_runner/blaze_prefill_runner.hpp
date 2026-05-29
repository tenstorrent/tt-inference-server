// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>

#include "config/runner_config.hpp"
#include "ipc/interface/result_queue.hpp"
#include "ipc/interface/task_queue.hpp"
#include "runtime/runners/blaze_prefill_runner/i_blaze_prefill_model_runner.hpp"
#include "runtime/runners/ipc_runner.hpp"
#include "services/memory_services/memory_manager.hpp"

namespace tt::runners {

class BlazePrefillRunner : public IRunner {
 public:
  BlazePrefillRunner(const tt::config::LLMConfig& config,
                     ipc::IResultQueue* resultQueue,
                     tt::ipc::ITaskQueue* taskQueue);
  ~BlazePrefillRunner() override;

  void run() override;
  void stop() override;
  bool warmup() override;
  const char* runnerType() const override { return "BlazePrefillRunner"; }

 private:
  void drainMemoryRequests();

  tt::config::LLMConfig config;
  ipc::IResultQueue* resultQueue;
  tt::ipc::ITaskQueue* taskQueue;
  std::unique_ptr<blaze_prefill::IBlazePrefillModelRunner> modelRunner;
  std::unique_ptr<tt::services::MemoryManager> memoryManager;
  std::atomic<bool> stopped{false};
  uint32_t nextSlotId{0};
};

}  // namespace tt::runners
