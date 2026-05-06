// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "config/defaults.hpp"
#include "config/types.hpp"

namespace tt::config {

/**
 * Configuration for LLM inference engine.
 * Includes model runner settings, scheduling policy, and KV cache parameters.
 */
struct LLMConfig {
  size_t max_num_batched_tokens = 64 * defaults::MAX_INPUT_TOKENS;
  size_t max_in_flight_count = 64;
  std::vector<int64_t> stop_token_ids;  // Set by tt::config::llmEngineConfig()
                                        // from active tokenizer strategy
  int eos = 1;
  size_t kvcache_block_size = 256;
  size_t num_kvcache_blocks = 512;
  ModelRunnerType runner_type = ModelRunnerType::MOCK;
  SchedulingPolicy scheduling_policy = SchedulingPolicy::PREFILL_FIRST;
};

/**
 * Configuration for embedding service.
 * Currently a placeholder - will be expanded as embedding features are added.
 */
struct EmbeddingConfig {
  ModelRunnerType runner_type = ModelRunnerType::MOCK;
};

/**
 * Variant wrapper for all runner configuration types.
 * Allows generic handling of different service configurations.
 */
using RunnerConfig = std::variant<LLMConfig, EmbeddingConfig>;

}  // namespace tt::config
