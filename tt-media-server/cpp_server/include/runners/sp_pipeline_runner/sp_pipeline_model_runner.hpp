// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "runners/llm_runner/sequence.hpp"
#include "runners/sp_pipeline_runner/shared_memory.hpp"
#include "utils/concurrent_queue.hpp"

namespace sp_pipeline {

using DecodeCallback = std::function<void(const llm_engine::TokenResult&)>;
using DecodeQueue = ConcurrentQueue<llm_engine::TokenResult>;

class SpPipelineModelRunner {
 public:
  explicit SpPipelineModelRunner(DecodeCallback callback);
  ~SpPipelineModelRunner();

  SpPipelineModelRunner(const SpPipelineModelRunner&) = delete;
  SpPipelineModelRunner& operator=(const SpPipelineModelRunner&) = delete;

  void write(const std::string& task_id,
                     const std::vector<int64_t>& token_ids,
                     uint32_t max_tokens);
  void exit();

 private:
  struct ShmNames {
    ShmNames() {
      const char* c2p = std::getenv("TT_IPC_SHM_C2P");
      const char* p2c = std::getenv("TT_IPC_SHM_P2C");
      write = c2p ? std::string(c2p) : throw std::runtime_error("TT_IPC_SHM_C2P not set");
      read = p2c ? std::string(p2c) : throw std::runtime_error("TT_IPC_SHM_P2C not set");
    }
    std::string write;
    std::string read;
  };

  void reader_loop();

  DecodeCallback decode_callback_;
  ShmNames shm_names_;
  PrefillSharedMemory device_input_;
  DecodeSharedMemory device_output_;
  std::atomic<bool> stop_{false};
  std::thread reader_thread_;
};

}  // namespace sp_pipeline
