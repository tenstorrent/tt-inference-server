// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "config/runner_config.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/result_queue.hpp"
#include "pipeline_manager/pipeline_manager.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"
#include "runners/sp_pipeline_runner/blaze_utils.hpp"
#include "services/memory_services/async_memory_manager.hpp"

namespace tt::runners {

namespace pm = tt_blaze::pipeline_manager;

class BlazeRunner : public IRunner {
 public:
  BlazeRunner(const tt::config::LLMConfig& config,
              ipc::IResultQueue* resultQueue,
              tt::runners::llm_engine::ITaskQueue* taskQueue);
  ~BlazeRunner() override;

  void run() override;
  void stop() override;
  bool warmup() override;
  const char* runnerType() const override { return "BlazeRunner"; }

 private:
  void step();

  std::optional<pm::PMResponse> getResponse();
  std::optional<pm::OutputMessage> getOutput();
  inline std::optional<tt::domain::ManageMemoryTask> getMemoryRequest();
  inline void handleMemoryRequest(const tt::domain::ManageMemoryTask& request);
  inline void handleResponse(const pm::PMResponse& response);
  void handleOutput(const pm::OutputMessage& output);
  std::unique_ptr<tt::runners::llm_engine::Sequence> getRequest();
  void handleRequest(
      std::unique_ptr<tt::runners::llm_engine::Sequence> request);
  void evictSlot(uint32_t slotId);
  void checkOutputHang();

  tt::config::LLMConfig config;
  std::unordered_set<int64_t> stopTokenIds;
  ipc::IResultQueue* resultQueue;
  tt::runners::llm_engine::ITaskQueue* taskQueue;
  std::unique_ptr<pm::PipelineManager> pipelineManager;
  std::unordered_map<uint32_t, blaze_utils::SlotContext> slotContexts;
  std::atomic<bool> stopped{false};
  std::unique_ptr<tt::services::AsyncMemoryManager> memoryManager;
  std::chrono::steady_clock::time_point lastOutputTime;
  std::chrono::milliseconds outputHangTimeout;
};
}  // namespace tt::runners
