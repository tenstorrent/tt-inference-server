// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>

#include "config/runner_config.hpp"
#include "ipc/result_queue.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"
#include "runners/sp_prefill_runner/i_sp_prefill_model_runner.hpp"

namespace tt::runners {

class SpPrefillRunner : public IRunner {
 public:
  SpPrefillRunner(const tt::config::LLMConfig& config,
                  ipc::IResultQueue* resultQueue,
                  tt::runners::llm_engine::ITaskQueue* taskQueue);
  ~SpPrefillRunner() override;

  void run() override;
  void stop() override;
  bool warmup() override;
  const char* runnerType() const override { return "SpPrefillRunner"; }

 private:
  tt::config::LLMConfig config;
  ipc::IResultQueue* resultQueue;
  tt::runners::llm_engine::ITaskQueue* taskQueue;
  std::unique_ptr<sp_prefill::ISpPrefillModelRunner> modelRunner;
  std::atomic<bool> stopped{false};
};

}  // namespace tt::runners
