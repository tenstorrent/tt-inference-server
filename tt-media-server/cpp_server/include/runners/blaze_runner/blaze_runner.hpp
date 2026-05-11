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
#include "domain/llm/sequence.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/interface/result_queue.hpp"
#include "ipc/interface/task_queue.hpp"
#include "pipeline_manager/pipeline_manager.hpp"
#include "runners/blaze_runner/blaze_utils.hpp"
#include "runners/runner_interface.hpp"
#include "services/memory_services/memory_manager.hpp"

namespace tt::runners {

using namespace tt::domain::llm;

namespace pm = tt_blaze::pipeline_manager;

class BlazeRunner : public IRunner {
 public:
  BlazeRunner(const tt::config::LLMConfig& config,
              ipc::IResultQueue* resultQueue, tt::ipc::ITaskQueue* taskQueue);
  ~BlazeRunner() override;

  void run() override;
  void stop() override;
  bool warmup() override;
  const char* runnerType() const override { return "BlazeRunner"; }

 private:
  void step();

  void drainAndHandleMemoryResponses();
  void drainAndHandleOutputs();
  inline std::optional<tt::domain::ManageMemoryTask> getMemoryRequest();
  inline void handleMemoryRequest(const tt::domain::ManageMemoryTask& request);
  inline void handleMemoryResponse(const pm::PMResponse& response);
  void handleOutput(const pm::OutputMessage& output);
  std::unique_ptr<tt::domain::llm::Sequence> getRequest();
  void handleRequest(std::unique_ptr<tt::domain::llm::Sequence> request);
  void evictSlot(uint32_t slotId);
  void checkOutputHang();

  tt::config::LLMConfig config;
  std::unordered_set<int64_t> stopTokenIds;
  ipc::IResultQueue* resultQueue;
  tt::ipc::ITaskQueue* taskQueue;
  std::unique_ptr<tt::domain::llm::Sequence> requestToRetry;
  std::unique_ptr<pm::PipelineManager> pipelineManager;
  std::unordered_map<uint32_t, blaze_utils::SlotContext> slotContexts;
  std::atomic<bool> stopped{false};
  std::unique_ptr<tt::services::MemoryManager> memoryManager;
  std::chrono::steady_clock::time_point lastOutputTime;
  std::chrono::milliseconds outputHangTimeout;
};
}  // namespace tt::runners
