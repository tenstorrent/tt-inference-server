// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "runners/sp_pipeline_runner/shared_memory.hpp"
#include "runners/sp_prefill_runner/i_sp_prefill_model_runner.hpp"

namespace sp_prefill {

using PrefillCallback = std::function<void(const llm_engine::TokenResult&)>;

class SpPrefillModelRunner : public ISpPrefillModelRunner {
 public:
  explicit SpPrefillModelRunner(PrefillCallback callback);
  ~SpPrefillModelRunner() override;

  SpPrefillModelRunner(const SpPrefillModelRunner&) = delete;
  SpPrefillModelRunner& operator=(const SpPrefillModelRunner&) = delete;

  void write(const std::string& taskId,
             const std::vector<int64_t>& tokenIds) override;
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

  PrefillCallback prefillCallback;
  ShmNames shmNames;
  sp_pipeline::PrefillSharedMemory deviceInput;
  sp_pipeline::DecodeSharedMemory deviceOutput;
  std::atomic<bool> stop{false};
  std::thread readerThread;
};

}  // namespace sp_prefill
