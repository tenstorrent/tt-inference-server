// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "utils/concurrent_queue.hpp"

namespace tt::runners::sp_pipeline {

using DecodeCallback =
    std::function<void(const tt::runners::llm_engine::TokenResult&)>;
using DecodeQueue =
    tt::utils::LockFreeSPSCQueue<tt::runners::llm_engine::TokenResult>;

enum class RequestPhase { PREFILL, DECODE };

class ISpPipelineModelRunner {
 public:
  virtual ~ISpPipelineModelRunner() = default;

  virtual void write(uint32_t taskId, const std::vector<int64_t>& tokenIds,
                     uint32_t maxTokens, RequestPhase phase,
                     bool fastMode = false) = 0;
  virtual void exit() = 0;
};

std::unique_ptr<ISpPipelineModelRunner> makeModelRunner(
    const tt::config::LLMConfig& config, DecodeCallback callback);

}  // namespace tt::runners::sp_pipeline
