// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <pipeline_manager/pipeline_interface.hpp>
#include <pipeline_manager/pipeline_manager.hpp>
#include <unordered_map>
#include <unordered_set>
#include "ipc/shared_memory.hpp"

#include "config/runner_config.hpp"
<<<<<<< HEAD
=======
#include "ipc/boost_ipc_memory_queue.hpp"
#include "ipc/token_ring_buffer.hpp"
>>>>>>> dev
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"
#include "domain/manage_memory.hpp"

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
  void pushToken(const llm_engine::TaskID& taskId, uint64_t tokenId,
                 bool finished);
  void pushErrorToken(const llm_engine::TaskID& taskId);

  std::optional<tt_blaze::pipeline_manager::PMResponse> getResponse();
  std::optional<tt_blaze::pipeline_manager::OutputMessage> getOutput();
  inline std::optional<tt::domain::ManageMemoryTask> getMemoryRequest();
  inline void handleMemoryRequest(const tt::domain::ManageMemoryTask& request);
  inline void handleResponse(const tt_blaze::pipeline_manager::PMResponse& response);
  void handleOutput(const tt_blaze::pipeline_manager::OutputMessage& output);
  std::unique_ptr<llm_engine::Sequence> getRequest();
  void handleRequest(std::unique_ptr<llm_engine::Sequence> request);
  void evictSlot(uint32_t slotId);

  tt::config::LLMConfig config;
  std::unordered_set<int64_t> stopTokenIds;
  ipc::TokenRingBuffer<65536>* resultQueue;
  llm_engine::ITaskQueue* taskQueue;
  std::unique_ptr<tt_blaze::pipeline_manager::PipelineManager> pipelineManager;
  std::unordered_map<uint32_t, std::unique_ptr<llm_engine::Sequence>> running;
  std::unordered_map<uint32_t, std::unique_ptr<llm_engine::Sequence>> allocating;
  std::atomic<bool> stopped{false};
  uint32_t nextRequestID{0};
};
}  // namespace tt::runners
