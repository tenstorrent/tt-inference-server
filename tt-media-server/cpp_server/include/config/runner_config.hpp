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

struct RunnerConfigBase {
  ModelRunnerType runner_type = ModelRunnerType::MOCK;
};

/** Shared fields for media runners (image, audio, video). Mirrors the
 *  device/weight knobs from tt-media-server's `config/settings.py`. */
struct MediaRunnerConfigBase : RunnerConfigBase {
  size_t worker_id = 0;
  size_t max_batch_size = 1;
  // 2-D {rows, cols}. rows > 1 enables tensor parallelism.
  std::vector<size_t> device_mesh_shape{1, 1};
  bool is_galaxy = false;
  // Device type string (e.g. "galaxy", "bh-galaxy", "n150").
  std::string device;
  // Empty = use the HF Hub default repo for the active runner.
  std::string model_weights_path;
  unsigned weights_distribution_timeout_seconds = 1800;
  std::string visible_devices;
};

// Config for the blaze decode/prefill runners. Carries the scheduler/pipeline
// knobs the runners, scheduler factory, and blaze_utils need, populated from
// the env-backed `tt::config::` accessors by `blazeConfig()`. Consumers read
// these fields instead of reaching into the global accessors directly.
struct BlazeConfig : RunnerConfigBase {
  // Sizing & timeouts
  size_t maxUsers = defaults::PM_MAX_USERS;
  unsigned warmupTimeoutMs = defaults::WARMUP_TIMEOUT_MS;
  unsigned outputHangTimeoutMs = defaults::OUTPUT_HANG_TIMEOUT_MS;

  // Scheduler params (decode + prefill)
  uint32_t modelNumLayers = defaults::MODEL_NUM_LAYERS;
  uint32_t prefillChunkSize = defaults::PREFILL_CHUNK_SIZE;
  bool enableMigration = defaults::ENABLE_MIGRATION;
  uint32_t migrationPrefillEndpointId =
      defaults::MIGRATION_PREFILL_ENDPOINT_ID;
  uint32_t migrationDecodeEndpointId =
      defaults::MIGRATION_DECODE_ENDPOINT_ID;
  std::string specDecodeMode = defaults::SPEC_DECODE_MODE;
  size_t mtpLevel = defaults::MTP_LEVEL;
  bool useMockScheduler = defaults::MOCK_USE_SCHEDULER;

  // Pipeline / channel config
  std::string blazeSocketDescriptorPrefix;
  unsigned pmConnectTimeoutMs = defaults::PM_CONNECT_TIMEOUT_MS;
  std::string wireFormat = defaults::WIRE_FORMAT;
  std::string prefillAckChannelName = defaults::PREFILL_ACK_CHANNEL_NAME;
  std::string migrationCmdQueueName = defaults::MIGRATION_CMD_QUEUE_NAME;
  std::string migrationTableQueueName = defaults::MIGRATION_TABLE_QUEUE_NAME;
  std::string migrationRespQueueName = defaults::MIGRATION_RESP_QUEUE_NAME;

  // Mock scheduler knobs
  unsigned mockPrefillLatencyMs = defaults::MOCK_PREFILL_CHUNK_LATENCY_MS;
  unsigned mockDecodeTokenLatencyUs = defaults::MOCK_DECODE_TOKEN_LATENCY_US;
  uint32_t mockDecodeTokenId = defaults::MOCK_DECODE_TOKEN_ID;

  // Generation fallbacks read by blaze_utils
  size_t maxContextLength = defaults::MAX_CONTEXT_LENGTH;
  bool sampleOnlyInReasoning = false;
};

struct EmbeddingConfig : RunnerConfigBase {};

struct ImageConfig : MediaRunnerConfigBase {
  ImageConfig() { runner_type = ModelRunnerType::TT_SDXL_GENERATE; }

  size_t image_width = 1024;
  size_t image_height = 1024;
};

using RunnerConfig = std::variant<BlazeConfig, EmbeddingConfig, ImageConfig>;

}  // namespace tt::config
