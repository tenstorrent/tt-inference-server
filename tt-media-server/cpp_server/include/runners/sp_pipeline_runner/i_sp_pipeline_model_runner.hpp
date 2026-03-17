// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "runners/llm_runner/sequence.hpp"
#include "utils/concurrent_queue.hpp"

namespace sp_pipeline {

using DecodeCallback = std::function<void(const llm_engine::TokenResult&)>;
using DecodeQueue = ConcurrentQueue<llm_engine::TokenResult>;

enum class RequestPhase { PREFILL, DECODE };

class ISpPipelineModelRunner {
 public:
  virtual ~ISpPipelineModelRunner() = default;

  virtual void write(const std::string& task_id,
                     const std::vector<int64_t>& token_ids,
                     uint32_t max_tokens,
                     RequestPhase phase) = 0;
  virtual void exit() = 0;
};

/// Factory: given a DecodeCallback, produce a model runner.
using ModelRunnerFactory = std::function<
    std::unique_ptr<ISpPipelineModelRunner>(DecodeCallback)>;

}  // namespace sp_pipeline
