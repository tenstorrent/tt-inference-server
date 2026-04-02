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
#include "ipc/boost_ipc_queue.hpp"
#include "ipc/token_ring_buffer.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"
#include "runners/sp_pipeline_runner/i_sp_pipeline_model_runner.hpp"

namespace tt::services {
class MemoryManager;
}

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
  void drainDecodeResults();
  void memoryLoop();
  void pushToken(uint32_t taskId, uint64_t tokenId, bool finished);
  void pushErrorToken(uint32_t taskId);

  tt::config::LLMConfig config;
  std::unordered_set<int64_t> stopTokenIds;
  ipc::TokenRingBuffer<65536>* resultQueue;
  llm_engine::ITaskQueue* taskQueue;
  std::unique_ptr<sp_pipeline::ISpPipelineModelRunner> modelRunner;
  sp_pipeline::DecodeQueue decodeQueue;
  std::unordered_map<uint32_t, std::unique_ptr<llm_engine::Sequence>>
      activeSequences;
  std::atomic<bool> stopped{false};
  int maxInFlightCount;
  int inFlightCount = 0;

  std::unique_ptr<tt::services::MemoryManager> memoryManager;
  std::thread memoryThread;
};

}  // namespace tt::runners
