// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "ipc/slot_ring_buffer.hpp"
#include "runners/sp_pipeline_runner/i_sp_pipeline_model_runner.hpp"

namespace tt::runners::sp_pipeline {

using DecodeCallback = std::function<void(const llm_engine::TokenResult&)>;

class SpPipelineModelRunner : public ISpPipelineModelRunner {
 public:
  explicit SpPipelineModelRunner(DecodeCallback callback);
  ~SpPipelineModelRunner() override;

  SpPipelineModelRunner(const SpPipelineModelRunner&) = delete;
  SpPipelineModelRunner& operator=(const SpPipelineModelRunner&) = delete;

  void write(uint32_t taskId, const std::vector<int64_t>& tokenIds,
             uint32_t maxTokens, RequestPhase phase, bool fastMode) override;
  void exit() override;

 private:
  struct ShmNames {
    ShmNames() {
      const char* c2p = std::getenv("TT_IPC_SHM_C2P");
      const char* p2c = std::getenv("TT_IPC_SHM_P2C");
      write = c2p ? std::string(c2p)
                  : throw std::runtime_error("TT_IPC_SHM_C2P not set");
      read = p2c ? std::string(p2c)
                 : throw std::runtime_error("TT_IPC_SHM_P2C not set");
    }
    std::string write;
    std::string read;
  };

  void readerLoop();

  DecodeCallback decodeCallback;
  ShmNames shmNames;
  tt::ipc::PrefillSlotBuffer deviceInput;
  tt::ipc::DecodeSlotBuffer deviceOutput;
  std::atomic<bool> stop{false};
  std::thread readerThread;
};

}  // namespace tt::runners::sp_pipeline
