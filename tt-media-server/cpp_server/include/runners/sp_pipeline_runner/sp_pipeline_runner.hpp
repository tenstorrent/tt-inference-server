// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "config/runner_config.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/token_ring_buffer.hpp"
#include "pipeline_manager/pipeline_manager.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"
#include "services/sp_pipeline_memory_manager.hpp"
namespace tt::runners {

namespace pm = tt_blaze::pipeline_manager;

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
  void pushToken(uint32_t taskId, uint64_t tokenId,
                 bool finished);
  void pushErrorToken(uint32_t taskId);

  std::optional<pm::PMResponse> getResponse();
  std::optional<pm::OutputMessage> getOutput();
  inline std::optional<tt::domain::ManageMemoryTask> getMemoryRequest();
  inline void handleMemoryRequest(const tt::domain::ManageMemoryTask& request);
  inline void handleResponse(const pm::PMResponse& response);
  void handleOutput(const pm::OutputMessage& output);
  std::unique_ptr<llm_engine::Sequence> getRequest();
  void handleRequest(std::unique_ptr<llm_engine::Sequence> request);
  void evictSlot(uint32_t slotId);

  tt::config::LLMConfig config;
  std::unordered_set<int64_t> stopTokenIds;
  ipc::TokenRingBuffer<65536>* resultQueue;
  llm_engine::ITaskQueue* taskQueue;
  std::unique_ptr<pm::PipelineManager> pipelineManager;
  std::unordered_map<uint32_t, std::unique_ptr<llm_engine::Sequence>> running;
  std::atomic<bool> stopped{false};
  std::unique_ptr<tt::services::SpPipelineMemoryManager> memoryManager;
};
}  // namespace tt::runners
