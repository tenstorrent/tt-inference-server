// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>

#include "config/runner_config.hpp"
#include "domain/sequence.hpp"
#include "ipc/result_queue.hpp"
#include "ipc/task_queue.hpp"
#include "runners/runner_interface.hpp"
#include "runners/sp_prefill_runner/i_blaze_prefill_model_runner.hpp"

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
  tt::config::LLMConfig config;
  ipc::IResultQueue* resultQueue;
  tt::ipc::ITaskQueue* taskQueue;
  std::unique_ptr<blaze_prefill::IBlazePrefillModelRunner> modelRunner;
  std::atomic<bool> stopped{false};
};

}  // namespace tt::runners
