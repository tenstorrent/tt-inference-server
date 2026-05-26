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
#include "runtime/runners/blaze_runner/blaze_slot_manager.hpp"
#include "runtime/runners/blaze_runner/blaze_types.hpp"
#include "runtime/runners/ipc_runner.hpp"
#include "services/memory_services/memory_manager.hpp"
#include "tt_llm_engine/scheduler/decode/decode_scheduler.hpp"
#include "tt_llm_engine/scheduler/decode/decode_types.hpp"

namespace tt::runners::blaze {

namespace ds = tt_llm_engine::scheduler::decode;

class BlazeRunner : public IRunner {
 public:
  BlazeRunner(
      const tt::config::LLMConfig& config, ipc::IResultQueue* resultQueue,
      tt::ipc::ITaskQueue* taskQueue, tt::ipc::ICancelQueue* cancelQueue,
      std::unique_ptr<tt::services::MemoryManager> memoryManager = nullptr);
  ~BlazeRunner() override;

  void run() override;
  void stop() override;
  bool warmup() override;
  const char* runnerType() const override { return "BlazeRunner"; }

 private:
  void step();

  void drainAndHandleMemoryResponses();
  void drainAndHandleOutputs();
  void drainAndHandleStopRequests();
  inline void handleStopRequest(uint32_t taskId);
  inline std::optional<tt::domain::ManageMemoryTask> getMemoryRequest();
  inline void handleMemoryRequest(const tt::domain::ManageMemoryTask& request);
  inline void handleAllocateRequest(
      const tt::domain::ManageMemoryTask& request);
  inline void handleEvictRequest(const tt::domain::ManageMemoryTask& request);
  inline void handleMemoryResponse(const ds::SchedulerResponse& response);
  inline void handleAllocateAck(uint32_t taskId, uint32_t slotId);
  inline void handleStopAck(uint32_t taskId, uint32_t slotId);
  inline void handleEvictAck(uint32_t taskId, uint32_t slotId);
  inline SlotContext* validateAck(uint32_t taskId, uint32_t slotId,
                                  const char* ackName);
  inline void handleDeferred(SlotContext& slot);
  void handleOutput(const ds::OutputMessage& output);
  std::unique_ptr<tt::domain::llm::Sequence> getRequest();
  void handleRequest(std::unique_ptr<tt::domain::llm::Sequence> request);
  void checkOutputHang();

  tt::config::LLMConfig config;
  ipc::IResultQueue* resultQueue;
  tt::ipc::ITaskQueue* taskQueue;
  tt::ipc::ICancelQueue* stopQueue;
  std::unique_ptr<ds::DecodeScheduler> decodeScheduler;
  PendingRequests pendingRequests;
  SlotManager slotManager;
  std::atomic<bool> stopped{false};
  std::unique_ptr<tt::services::MemoryManager> memoryManager;
  std::chrono::steady_clock::time_point lastOutputTime;
  std::chrono::milliseconds outputHangTimeout;
};
}  // namespace tt::runners::blaze
