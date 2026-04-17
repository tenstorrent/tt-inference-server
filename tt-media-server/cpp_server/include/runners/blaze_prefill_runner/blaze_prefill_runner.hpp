// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>

#include "config/runner_config.hpp"
#include "ipc/result_queue.hpp"
#include "runners/blaze_prefill_runner/i_blaze_prefill_model_runner.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"

namespace tt::runners {

class BlazePrefillRunner : public IRunner {
 public:
  BlazePrefillRunner(const tt::config::LLMConfig& config,
                     ipc::IResultQueue* resultQueue,
                     tt::runners::llm_engine::ITaskQueue* taskQueue);
  ~BlazePrefillRunner() override;

  void run() override;
  void stop() override;
  bool warmup() override;
  const char* runnerType() const override { return "BlazePrefillRunner"; }

 private:
  tt::config::LLMConfig config;
  ipc::IResultQueue* resultQueue;
  tt::runners::llm_engine::ITaskQueue* taskQueue;
  std::unique_ptr<blaze_prefill::IBlazePrefillModelRunner> modelRunner;
  std::atomic<bool> stopped{false};
};

}  // namespace tt::runners
