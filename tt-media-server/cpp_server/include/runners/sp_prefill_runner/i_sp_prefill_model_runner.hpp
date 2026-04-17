// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "runners/llm_runner/sequence.hpp"

namespace sp_prefill {

class ISpPrefillModelRunner {
 public:
  virtual ~ISpPrefillModelRunner() = default;

  // Prefill runner always does prefill, returns the single result token
  // (nullopt if stopped before result arrives)
  virtual std::optional<tt::runners::llm_engine::TokenResult> forward(
      uint32_t taskId, const std::vector<int64_t>& tokenIds) = 0;
  virtual void exit() = 0;
};

std::unique_ptr<ISpPrefillModelRunner> makeModelRunner(
    const tt::config::LLMConfig& config);

}  // namespace sp_prefill
