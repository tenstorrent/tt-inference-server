// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "config/defaults.hpp"
#include "config/types.hpp"

namespace tt::config {

/** Carries the runner_type used for factory dispatch. */
struct RunnerConfigBase {
  ModelRunnerType runner_type = ModelRunnerType::MOCK;
};

/** Shared fields for in-process media runners (image today; audio, TTS,
 *  video next). Mirrors the device/weight knobs that live at the global
 *  Settings level in tt-media-server's config/settings.py. */
struct MediaRunnerConfigBase : RunnerConfigBase {
  size_t max_batch_size = 1;

  // 2-D {rows, cols}. rows > 1 enables tensor parallelism, mirroring
  // BaseDeviceRunner.is_tensor_parallel.
  std::vector<size_t> device_mesh_shape{1, 1};

  bool is_galaxy = false;

  // Empty = use the HF Hub default repo for the active runner.
  std::string model_weights_path;

  unsigned weights_distribution_timeout_seconds = 1800;
};

struct LLMConfig : RunnerConfigBase {
  size_t max_num_batched_tokens = 64 * defaults::MAX_CONTEXT_LENGTH;
  size_t max_in_flight_count = 64;
  std::vector<int64_t> stop_token_ids;  // Set by tt::config::llmEngineConfig()
                                        // from active tokenizer strategy
  int eos = 1;
  size_t kvcache_block_size = 256;
  size_t num_kvcache_blocks = 512;
  SchedulingPolicy scheduling_policy = SchedulingPolicy::PREFILL_FIRST;
};

/** Placeholder; will grow as embedding features are added. */
struct EmbeddingConfig : RunnerConfigBase {};

/** Per-request fields (prompt, num_inference_steps, ...) live on the request,
 *  not here. */
struct ImageConfig : MediaRunnerConfigBase {
  ImageConfig() { runner_type = ModelRunnerType::TT_SDXL_GENERATE; }

  size_t image_width = 1024;
  size_t image_height = 1024;
};

/** Variant wrapper for all runner configuration types. */
using RunnerConfig = std::variant<LLMConfig, EmbeddingConfig, ImageConfig>;

}  // namespace tt::config
