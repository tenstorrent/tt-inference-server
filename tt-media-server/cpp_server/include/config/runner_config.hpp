// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "config/types.hpp"

namespace tt::config {

/**
 * Configuration for LLM inference engine.
 * Includes model runner settings, scheduling policy, and KV cache parameters.
 */
struct LLMConfig {
  static constexpr size_t MAX_INPUT_TOKENS = 131072;  // 128k
  size_t max_num_batched_tokens = 64 * MAX_INPUT_TOKENS;
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
 * Configuration for image generation service. Per-model knobs only; per-request
 * fields (prompt, num_inference_steps, guidance, ...) live on the request.
 */
struct ImageConfig {
  ModelRunnerType runner_type = ModelRunnerType::TT_SDXL_GENERATE;

  size_t max_batch_size = 1;

  // 2-D {rows, cols}. rows > 1 enables tensor parallelism, mirroring
  // BaseDeviceRunner.is_tensor_parallel.
  std::vector<size_t> device_mesh_shape{1, 1};

  size_t image_width = 1024;
  size_t image_height = 1024;

  bool is_galaxy = false;

  // Empty = use the HF Hub default repo for the active runner.
  std::string model_weights_path;

  unsigned weights_distribution_timeout_seconds = 1800;
};

/** Variant wrapper for all runner configuration types. */
using RunnerConfig = std::variant<LLMConfig, EmbeddingConfig, ImageConfig>;

}  // namespace tt::config
