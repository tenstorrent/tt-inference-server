// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <unordered_set>

#include "config/runner_config.hpp"
#include "domain/llm/sequence.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/interface/cancel_queue.hpp"
#include "ipc/interface/result_queue.hpp"
#include "ipc/interface/task_queue.hpp"
#include "runtime/runners/blaze_runner/blaze_types.hpp"
#include "runtime/runners/ipc_runner.hpp"
#include "services/memory_services/blaze_memory_manager.hpp"
#include "tt_llm_engine/scheduler/decode/decode_scheduler.hpp"
#include "tt_llm_engine/scheduler/decode/decode_types.hpp"

namespace tt::runners {

using namespace tt::domain::llm;

namespace ds = tt_llm_engine::scheduler::decode;

class BlazeRunner : public IRunner {
 public:
  BlazeRunner(const tt::config::LLMConfig& config,
              ipc::IResultQueue* resultQueue, tt::ipc::ITaskQueue* taskQueue,
              tt::ipc::ICancelQueue* cancelQueue);
  ~BlazeRunner() override;

  void run() override;
  void stop() override;
  bool warmup() override;
  const char* runnerType() const override { return "BlazeRunner"; }

 private:
  void step();

  void drainAndHandleMemoryResponses();
  void drainAndHandleOutputs();
  void drainAndHandleCancelRequests();
  inline void handleCancelRequest(uint32_t taskId);
  inline std::optional<tt::domain::ManageMemoryTask> getMemoryRequest();
  inline void handleMemoryRequest(const tt::domain::ManageMemoryTask& request);
  inline void handleAllocateRequest(
      const tt::domain::ManageMemoryTask& request);
  inline void handleEvictRequest(const tt::domain::ManageMemoryTask& request);
  inline void handleMemoryResponse(const ds::SchedulerResponse& response);
  inline void handleAllocateAck(uint32_t taskId, uint32_t slotId);
  inline void handleStopAck(blaze_types::SlotContext& slot);
  inline void handleEvictAck(blaze_types::SlotContext& slot);
  inline void handleDeferred(blaze_types::SlotContext& slot);
  void handleOutput(const ds::OutputMessage& output);
  std::unique_ptr<tt::domain::llm::Sequence> getRequest();
  void handleRequest(std::unique_ptr<tt::domain::llm::Sequence> request);
  void checkOutputHang();

  tt::config::LLMConfig config;
  std::unordered_set<int64_t> stopTokenIds;
  ipc::IResultQueue* resultQueue;
  tt::ipc::ITaskQueue* taskQueue;
  tt::ipc::ICancelQueue* cancelQueue;
  std::unique_ptr<tt::domain::llm::Sequence> requestToRetry;
  std::unique_ptr<ds::DecodeScheduler> decodeScheduler;
  blaze_types::SlotManager slotManager;
  std::unordered_set<uint32_t> pendingAllocates;
  std::optional<tt::domain::ManageMemoryTask> pendingMemoryRetry;
  std::atomic<bool> stopped{false};
  std::unique_ptr<tt::services::BlazeMemoryManager> memoryManager;
  std::chrono::steady_clock::time_point lastOutputTime;
  std::chrono::milliseconds outputHangTimeout;
};
}  // namespace tt::runners
