// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "runners/runner_interface.hpp"
#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "ipc/shared_memory.hpp"

namespace tt::runners {

/**
 * Runner for DeepSeek on-device autoregressive decoding.
 *
 * 3-thread architecture:
 *   Main thread  – pops tasks from task_queue, tracks in-flight sequences
 *   Writer thread – (inside model_runner) writes prefills to device via C2P shm
 *   Reader thread – (inside model_runner) reads tokens from P2C shm, pushes to result_queue
 */
class DeepSeekRunner : public IRunner {
 public:
  DeepSeekRunner(const llm_engine::Config& config,
                 ipc::TokenRingBuffer<65536>* result_queue,
                 llm_engine::ITaskQueue* task_queue);
  ~DeepSeekRunner() override;

  void run() override;
  void stop() override;
  const char* runner_type() const override { return "DeepSeekRunner"; }

 private:
  void step();

  llm_engine::Config config_;
  ipc::TokenRingBuffer<65536>* result_queue_;
  llm_engine::ITaskQueue* task_queue_;
  std::unique_ptr<llm_engine::IModelRunner> model_runner_;

  struct InFlightSeq {
    std::string task_id_str;
    uint32_t max_tokens;
    uint32_t tokens_received;
    bool ignore_eos;
  };

  std::mutex in_flight_mutex_;
  std::unordered_map<llm_engine::TaskID, InFlightSeq> in_flight_;
  std::atomic<int> in_flight_count_{0};

  std::atomic<bool> stopped_{false};
};

}  // namespace tt::runners
