// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "ipc/slot_ring_buffer.hpp"
#include "runners/blaze_prefill_runner/i_blaze_prefill_model_runner.hpp"

namespace blaze_prefill {

class BlazePrefillModelRunner : public IBlazePrefillModelRunner {
 public:
  BlazePrefillModelRunner();
  ~BlazePrefillModelRunner() override;

  BlazePrefillModelRunner(const BlazePrefillModelRunner&) = delete;
  BlazePrefillModelRunner& operator=(const BlazePrefillModelRunner&) = delete;

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
  std::atomic<size_t> consecutiveErrors{0};

  static constexpr size_t MAX_CONSECUTIVE_ERRORS = 5;
};

}  // namespace blaze_prefill
