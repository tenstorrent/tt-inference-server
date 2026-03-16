// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "config/types.hpp"

namespace tt::config {

/**
 * Configuration for LLM inference engine.
 * Includes model runner settings, scheduling policy, and KV cache parameters.
 */
struct LLMConfig {
    static constexpr size_t MAX_INPUT_TOKENS = 131072;  // 128k
    int max_num_batched_tokens = 64 * MAX_INPUT_TOKENS;
    int max_in_flight_count = 64;
    std::vector<int64_t> stop_token_ids;  // Set by create_llm_config() from active tokenizer strategy
    int eos = 1;
    int kvcache_block_size = 256;
    int num_kvcache_blocks = 512;
    ModelRunnerType runner_type = ModelRunnerType::Mock;
    SchedulingPolicy scheduling_policy = SchedulingPolicy::PREFILL_FIRST;
};

/**
 * Factory function to create LLMConfig from environment variables and runtime settings.
 * Declared here, implemented in src/config/settings.cpp.
 */
LLMConfig create_llm_config();

}  // namespace tt::config
