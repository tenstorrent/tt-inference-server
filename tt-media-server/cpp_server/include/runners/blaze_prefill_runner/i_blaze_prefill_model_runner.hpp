// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/llm/sequence.hpp"

namespace blaze_prefill {

using namespace tt::domain::llm;

class IBlazePrefillModelRunner {
 public:
  virtual ~IBlazePrefillModelRunner() = default;

  // Prefill runner always does prefill, returns the single result token
  // (nullopt if stopped before result arrives)
  virtual std::optional<tt::domain::llm::TokenResult> forward(
      uint32_t taskId, const std::vector<int64_t>& tokenIds,
      uint32_t slotId) = 0;
  virtual void exit() = 0;
};

std::unique_ptr<IBlazePrefillModelRunner> makeModelRunner(
    const tt::config::LLMConfig& config);

}  // namespace blaze_prefill
