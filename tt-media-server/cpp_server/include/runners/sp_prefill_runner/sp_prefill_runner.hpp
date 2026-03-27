// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "config/runner_config.hpp"
#include "ipc/boost_ipc_memory_queue.hpp"
#include "ipc/shared_memory.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"
#include "runners/sp_prefill_runner/i_sp_prefill_model_runner.hpp"
#include "services/memory_manager.hpp"

namespace tt::runners {

class SpPrefillRunner : public IRunner {
 public:
  SpPrefillRunner(const tt::config::LLMConfig& config,
                  ipc::TokenRingBuffer<65536>* resultQueue,
                  llm_engine::ITaskQueue* taskQueue);
  ~SpPrefillRunner() override;

  void run() override;
  void stop() override;
  bool warmup();
  const char* runnerType() const override { return "SpPrefillRunner"; }

 private:
  void step();
  void drainPrefillResults();
  void memoryLoop();
  void pushToken(const llm_engine::TaskID& taskId, uint64_t tokenId,
                 bool finished);
  void pushErrorToken(const llm_engine::TaskID& taskId);

  tt::config::LLMConfig config;
  std::unordered_set<int64_t> stopTokenIds;
  ipc::TokenRingBuffer<65536>* resultQueue;
  llm_engine::ITaskQueue* taskQueue;
  std::unique_ptr<sp_prefill::ISpPrefillModelRunner> modelRunner;
  sp_prefill::PrefillQueue prefillQueue;
  std::unique_ptr<llm_engine::Sequence> activeSequence;
  std::atomic<bool> stopped{false};

  tt::services::MemoryManager memoryManager;
  ipc::MemoryRequestQueue memoryRequests{ipc::k_memory_request_queue_name,
                                         ipc::MEMORY_QUEUE_CAPACITY};
  ipc::MemoryResultQueue memoryResults{ipc::k_memory_result_queue_name,
                                       ipc::MEMORY_QUEUE_CAPACITY};
  std::thread memoryThread;
};

}  // namespace tt::runners
