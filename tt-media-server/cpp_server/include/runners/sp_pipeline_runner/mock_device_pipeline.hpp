// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "runners/llm_runner/sequence.hpp"
#include "runners/sp_pipeline_runner/i_sp_pipeline_model_runner.hpp"

namespace tt::runners::sp_pipeline {

struct MockDeviceConfig {
  size_t numStages = 64;
  size_t stageDurationUs = 44;
  size_t prefillChunkSize = 16;
  size_t writeQueueCapacity = 64;
};

/// Simulates a pipelined device where prefill feeds chunk_size tokens into
/// stage 0 (one per tick, blocking), and decode feeds 1 token per tick.
/// Chat completions fire when the last fed token exits the final pipeline
/// stage. Between chunks the scheduler round-robins: decode first, then
/// prefill.
class MockDevicePipeline {
 public:
  explicit MockDevicePipeline(MockDeviceConfig config = {});
  ~MockDevicePipeline();

  MockDevicePipeline(const MockDevicePipeline&) = delete;
  MockDevicePipeline& operator=(const MockDevicePipeline&) = delete;

  void write(uint32_t taskId, const std::vector<int64_t>& tokenIds,
             uint32_t maxTokens, RequestPhase phase);

  std::optional<tt::runners::llm_engine::TokenResult> read();

  void exit();

 private:
  struct PipelineRequest {
    uint32_t taskId;
    std::vector<int64_t> tokenIds;
    uint32_t maxTokens;
    uint32_t tokensGenerated = 0;
    uint32_t prefillOffset = 0;
    bool isDecode = false;
  };

  using RequestPtr = std::unique_ptr<PipelineRequest>;

  struct InFlightRequest {
    size_t completeAtTick;
    RequestPtr req;
  };

  void pipelineLoop();
  void drainInput();
  void emitToken(RequestPtr& req);
  void handleCompletion(RequestPtr req);
  void insertInFlight(InFlightRequest entry);
  void trySchedule();
  RequestPtr scheduleNext();

  MockDeviceConfig config;
  size_t currentTick = 0;

  // The request currently feeding tokens into stage 0.
  RequestPtr activeReq;
  size_t feedRemaining = 0;

  // Requests currently traversing the pipeline, sorted by exit tick.
  std::deque<InFlightRequest> inFlightPipeline;

  // Scheduling queues (pipeline thread only).
  std::deque<RequestPtr> decodeQueue;
  std::deque<RequestPtr> prefillQueue;

  // Bounded input queue — write() blocks when full.
  std::deque<RequestPtr> inputQueue;
  std::mutex inputMutex;
  std::condition_variable inputNotFull;

  // Output queue — read() blocks when empty.
  std::deque<tt::runners::llm_engine::TokenResult> outputQueue;
  std::mutex outputMutex;
  std::condition_variable outputNotEmpty;

  std::atomic<bool> stop{false};
  std::thread pipelineThread;
};

}  // namespace tt::runners::sp_pipeline
