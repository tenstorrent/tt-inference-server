// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "config/runner_config.hpp"
#include "ipc/shared_memory.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"
#include "runners/sp_pipeline_runner/i_sp_pipeline_model_runner.hpp"

namespace tt::runners {

class SpPipelineRunner : public IRunner {
 public:
  SpPipelineRunner(const tt::config::LLMConfig& config,
                   ipc::TokenRingBuffer<65536>* result_queue,
                   llm_engine::ITaskQueue* task_queue,
                   sp_pipeline::ModelRunnerFactory model_runner_factory);
  ~SpPipelineRunner() override;

  void run() override;
  void stop() override;
  bool warmup();
  const char* runner_type() const override { return "SpPipelineRunner"; }

 private:
  void step();
  void drain_decode_results();
  void push_token(const llm_engine::TaskID& task_id, uint64_t token_id,
                  bool finished);
  void push_error_token(const llm_engine::TaskID& task_id);

  tt::config::LLMConfig config_;
  std::unordered_set<int64_t> stop_token_ids_;
  ipc::TokenRingBuffer<65536>* result_queue_;
  llm_engine::ITaskQueue* task_queue_;
  std::unique_ptr<sp_pipeline::ISpPipelineModelRunner> model_runner_;
  sp_pipeline::DecodeQueue decode_queue_;
  std::unordered_map<llm_engine::TaskID, std::unique_ptr<llm_engine::Sequence>>
      active_sequences_;
  std::atomic<bool> stopped_{false};
  int max_in_flight_count_;
  int in_flight_count_ = 0;
};

}  // namespace tt::runners
