// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "ipc/slot_ring_buffer.hpp"
#include "runners/sp_prefill_runner/i_sp_prefill_model_runner.hpp"

namespace sp_prefill {

class SpPrefillModelRunner : public ISpPrefillModelRunner {
 public:
  SpPrefillModelRunner();
  ~SpPrefillModelRunner() override;

  SpPrefillModelRunner(const SpPrefillModelRunner&) = delete;
  SpPrefillModelRunner& operator=(const SpPrefillModelRunner&) = delete;

  std::optional<tt::runners::llm_engine::TokenResult> forward(
      uint32_t taskId, const std::vector<int64_t>& tokenIds) override;
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

  ShmNames shmNames;
  tt::ipc::PrefillSlotBuffer deviceInput;
  tt::ipc::DecodeSlotBuffer deviceOutput;
  std::atomic<bool> stop{false};
};

}  // namespace sp_prefill
