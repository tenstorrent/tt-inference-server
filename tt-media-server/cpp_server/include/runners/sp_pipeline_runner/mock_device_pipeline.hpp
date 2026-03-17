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

namespace sp_pipeline {

struct MockDeviceConfig {
  size_t num_stages = 64;
  size_t stage_duration_us = 44;
  size_t prefill_chunk_size = 16;
  size_t write_queue_capacity = 64;
};

/// Simulates a pipelined device where prefill feeds chunk_size tokens into
/// stage 0 (one per tick, blocking), and decode feeds 1 token per tick.
/// Completions fire when the last fed token exits the final pipeline stage.
/// Between chunks the scheduler round-robins: decode first, then prefill.
class MockDevicePipeline {
 public:
  explicit MockDevicePipeline(MockDeviceConfig config = {});
  ~MockDevicePipeline();

  MockDevicePipeline(const MockDevicePipeline&) = delete;
  MockDevicePipeline& operator=(const MockDevicePipeline&) = delete;

  void write(const std::string& task_id,
             const std::vector<int64_t>& token_ids,
             uint32_t max_tokens,
             RequestPhase phase);

  std::optional<llm_engine::TokenResult> read();

  void exit();

 private:
  struct PipelineRequest {
    std::string task_id;
    std::vector<int64_t> token_ids;
    uint32_t max_tokens;
    uint32_t tokens_generated = 0;
    uint32_t prefill_offset = 0;
    bool is_decode = false;
  };

  using RequestPtr = std::unique_ptr<PipelineRequest>;

  struct PendingCompletion {
    size_t complete_at_tick;
    RequestPtr req;
  };

  void pipeline_loop();
  void drain_input();
  void handle_completion(RequestPtr req);
  void insert_completion(PendingCompletion completion);
  void try_schedule();
  RequestPtr schedule_next();

  MockDeviceConfig config_;
  size_t current_tick_ = 0;

  // The request currently feeding tokens into stage 0.
  RequestPtr active_req_;
  size_t feed_remaining_ = 0;

  // Completions ordered by tick (always naturally sorted).
  std::deque<PendingCompletion> pending_completions_;

  // Scheduling queues (pipeline thread only).
  std::deque<RequestPtr> decode_queue_;
  std::deque<RequestPtr> prefill_queue_;

  // Bounded input queue — write() blocks when full.
  std::deque<RequestPtr> input_queue_;
  std::mutex input_mutex_;
  std::condition_variable input_not_full_;

  // Output queue — read() blocks when empty.
  std::deque<llm_engine::TokenResult> output_queue_;
  std::mutex output_mutex_;
  std::condition_variable output_not_empty_;

  std::atomic<bool> stop_{false};
  std::thread pipeline_thread_;
};

}  // namespace sp_pipeline
