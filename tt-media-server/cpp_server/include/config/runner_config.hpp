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
 * Configuration for image generation service.
 * Mirrors the per-model knobs read by Python image runners
 * (resolution, batch, mesh shape, weights path). Per-request fields
 * (prompt, num_inference_steps, guidance, ...) live on the request struct.
 */
struct ImageConfig {
  // Default to SDXL Generate; image service has no mock runner since the
  // real SDXL runners now propagate clean ModuleNotFoundError messages
  // when ttnn / diffusers / tt-metal aren't on PYTHONPATH.
  ModelRunnerType runner_type = ModelRunnerType::TT_SDXL_GENERATE;

  // Batch / mesh
  size_t max_batch_size = 1;
  // 2-D mesh shape {rows, cols}, e.g. {1, 1} = single device,
  // {2, 4} = 2 TP shards x 4 replicas (8 devices).
  // Index 0 (rows) controls tensor-parallel width — TP is enabled iff
  // device_mesh_shape[0] > 1, mirroring BaseDeviceRunner.is_tensor_parallel.
  // Index 1 (cols) is the per-replica axis.
  std::vector<size_t> device_mesh_shape{1, 1};

  // Resolution (parsed from SDXL_IMAGE_RESOLUTION env, e.g. "1024x1024").
  size_t image_width = 1024;
  size_t image_height = 1024;

  // Galaxy / multi-host fabric flag, mirroring settings.is_galaxy.
  bool is_galaxy = false;

  // Path to local model weights. Empty means "use HF Hub default" — the
  // SDXL runners then pull from SupportedModels.STABLE_DIFFUSION_XL_BASE etc.
  std::string model_weights_path;

  // How long the SDXL runner allows the on-device weight distribution
  // (TtSDXLPipeline construction) to take before failing warmup.
  unsigned weights_distribution_timeout_seconds = 1800;
};

/**
 * Variant wrapper for all runner configuration types.
 * Allows generic handling of different service configurations.
 */
using RunnerConfig = std::variant<LLMConfig, EmbeddingConfig, ImageConfig>;

}  // namespace tt::config
