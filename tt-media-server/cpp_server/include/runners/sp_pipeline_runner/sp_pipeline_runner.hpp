// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
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
#include "services/memory_manager.hpp"
#include <pipeline_manager/pipeline_manager.hpp>
#include <pipeline_manager/pipeline_interface.hpp>

namespace tt::runners {

class SpPipelineRunner : public IRunner {
 public:
  SpPipelineRunner(const tt::config::LLMConfig& config,
                   ipc::TokenRingBuffer<65536>* resultQueue,
                   llm_engine::ITaskQueue* taskQueue);
  ~SpPipelineRunner() override;

  void run() override;
  void stop() override;
  bool warmup();
  const char* runnerType() const override { return "SpPipelineRunner"; }

 private:
  void step();
  void memoryLoop();
  void pushToken(const llm_engine::TaskID& taskId, uint64_t tokenId,
                 bool finished);
  void pushErrorToken(const llm_engine::TaskID& taskId);
  
  std::optional<tt_blaze::pipeline_manager::PMResponse> getResponse();
  std::optional<tt_blaze::pipeline_manager::OutputMessage> getOutput();
  void handleResponse(tt_blaze::pipeline_manager::PMResponse& response);
  void handleOutput(tt_blaze::pipeline_manager::OutputMessage& output);
  std::unique_ptr<llm_engine::Sequence> getRequest();
  void handleRequest(std::unique_ptr<llm_engine::Sequence> request);

  tt::config::LLMConfig config;
  std::unordered_set<int64_t> stopTokenIds;
  ipc::TokenRingBuffer<65536>* resultQueue;
  llm_engine::ITaskQueue* taskQueue;
  std::unique_ptr<tt_blaze::pipeline_manager::PipelineInterface> pipeline;
  std::unique_ptr<tt_blaze::pipeline_manager::PipelineManager> pipelineManager;
  std::unordered_map<uint32_t, std::unique_ptr<llm_engine::Sequence>> slotToSequence;
  std::unordered_map<uint32_t, std::unique_ptr<llm_engine::Sequence>> requestToSequence;
  std::atomic<bool> stopped{false};
  int maxInFlightCount;
  int inFlightCount = 0;

  tt::services::MemoryManager memoryManager;
  ipc::MemoryRequestQueue memoryRequests{ipc::k_memory_request_queue_name,
                                         ipc::MEMORY_QUEUE_CAPACITY};
  ipc::MemoryResultQueue memoryResults{ipc::k_memory_result_queue_name,
                                       ipc::MEMORY_QUEUE_CAPACITY};
  std::thread memoryThread;
  uint32_t nextRequestID {0};
};

}  // namespace tt::runners
