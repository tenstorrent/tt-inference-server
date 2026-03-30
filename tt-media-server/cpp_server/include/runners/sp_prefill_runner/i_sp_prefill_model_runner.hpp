// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "utils/concurrent_queue.hpp"

namespace sp_prefill {

using PrefillCallback = std::function<void(const llm_engine::TokenResult&)>;
using PrefillQueue = LockFreeConcurrentQueue<llm_engine::TokenResult>;

class ISpPrefillModelRunner {
 public:
  virtual ~ISpPrefillModelRunner() = default;

  // Prefill runner always does prefill, returns one token
  virtual void write(const std::string& taskId,
                     const std::vector<int64_t>& tokenIds) = 0;
  virtual void exit() = 0;
};

std::unique_ptr<ISpPrefillModelRunner> makeModelRunner(
    const tt::config::LLMConfig& config, PrefillCallback callback);

}  // namespace sp_prefill
