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

size_t maxContextLength();

struct RunnerConfigBase {
  ModelRunnerType runner_type = ModelRunnerType::MOCK;
};

/** Shared fields for media runners (image, TTS). Mirrors the
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
  // Route the PrefillScheduler's cross-endpoint (P->D) KV migration through
  // the Kafka-backed RemoteKVManagerAdapter (composed with a shmem/mock
  // loopback for migrate() calls the adapter cannot service). Only effective
  // when enableMigration is also true and the binary was built with
  // KAFKA_ENABLED=ON; toggling it on a non-Kafka build fails loudly at
  // scheduler construction rather than silently downgrading to the shmem
  // path. See makeMigrationClientInterface in blaze_utils.hpp.
  bool prefillUseRemoteKvManager = defaults::PREFILL_USE_REMOTE_KV_MANAGER;
  // Transport selector for the RemoteKVManager path: "kafka" (existing
  // per-worker Kafka fan-out) or "zmq" (single kv_manager prefill-leader
  // endpoint; kv_manager fans out internally). Only consulted when
  // prefillUseRemoteKvManager is true. See makeMigrationClientInterface in
  // blaze_utils.hpp for the dispatch logic.
  std::string prefillKvManagerTransport =
      defaults::PREFILL_KV_MANAGER_TRANSPORT;
  // ZMQ endpoints for the kv_manager prefill-leader control channel. Both
  // are ZMQ URIs (e.g. "tcp://0.0.0.0:5559"); this side binds and
  // kv_manager connects.
  std::string kvmZmqCmdEndpoint = defaults::KVM_ZMQ_CMD_ENDPOINT;
  std::string kvmZmqReplyEndpoint = defaults::KVM_ZMQ_REPLY_ENDPOINT;
  std::string kvmZmqTopic = defaults::KVM_ZMQ_TOPIC;
  uint32_t migrationPrefillEndpointId = defaults::MIGRATION_PREFILL_ENDPOINT_ID;
  uint32_t migrationDecodeEndpointId = defaults::MIGRATION_DECODE_ENDPOINT_ID;
  std::string specDecodeMode = defaults::SPEC_DECODE_MODE;
  size_t specLevel = defaults::SPEC_LEVEL;
  uint32_t blazeNumberOfPipelineStages =
      defaults::BLAZE_NUMBER_OF_PIPELINE_STAGES;

  // Pipeline / channel config
  std::string blazeSocketDescriptorPrefix;
  unsigned pmConnectTimeoutMs = defaults::PM_CONNECT_TIMEOUT_MS;
  std::string wireFormat = defaults::WIRE_FORMAT;
  std::string prefillAckChannelName = defaults::PREFILL_ACK_CHANNEL_NAME;
  std::string migrationCmdQueueName = defaults::MIGRATION_CMD_QUEUE_NAME;
  std::string migrationTableQueueName = defaults::MIGRATION_TABLE_QUEUE_NAME;
  std::string migrationRespQueueName = defaults::MIGRATION_RESP_QUEUE_NAME;

  // Mock pipeline knobs
  unsigned numPipelineStages = defaults::MOCK_PIPELINE_STAGES;
  unsigned mockStageLatencyUs = defaults::MOCK_STAGE_LATENCY_US;
  unsigned mockPrefillLatencyMs = defaults::MOCK_PREFILL_CHUNK_LATENCY_MS;
  unsigned mockDecodeTokenId = defaults::MOCK_DECODE_TOKEN_ID;

  // Generation fallbacks read by blaze_utils
  size_t maxContextLength = defaults::MAX_CONTEXT_LENGTH;
};

/** Config for the embedding runners. Deliberately standalone rather than
 *  derived from MediaRunnerConfigBase: the embedding path reads none of the
 *  image/TTS weight-distribution fields, and inheriting would make it grow
 *  whenever those gain a field. `device_mesh_shape` is likewise absent -
 *  nothing here reads it, and Python derives the mesh from its own
 *  (runner, device) table in config/constants.py. */
struct EmbeddingConfig : RunnerConfigBase {
  EmbeddingConfig() { runner_type = ModelRunnerType::TT_BGE_LARGE_EN; }

  size_t worker_id = 0;
  // Chip ids this worker may use, e.g. "0" or "0,1" (from DEVICE_IDS).
  std::string visible_devices;
  // Hard cap on requests per forward pass. Python derives it from its own
  // (MODEL, DEVICE) table; the parent process needs it too because its
  // dispatch thread forms the batches, so the worker verifies the two agree
  // once Python is up.
  size_t max_batch_size = 1;
  // Device type string, e.g. "n150" - selects the Python model config row.
  std::string device;

  // The two names C++ must know before Python exists. hf_model_id is what
  // clients send and what the runner validates against; python_model_name is
  // the internal enum value exported as MODEL, without which Python's Settings
  // skips its config lookup entirely. Which Python class implements the model
  // is not here: tt_model_runners/runner_fabric.py picks it from MODEL_RUNNER.
  std::string hf_model_id;
  std::string python_model_name;
};

struct ImageConfig : MediaRunnerConfigBase {
  ImageConfig() { runner_type = ModelRunnerType::TT_SDXL_GENERATE; }

  size_t imageWidth = 1024;
  size_t imageHeight = 1024;
};

struct TtsConfig : RunnerConfigBase {
  TtsConfig() { runner_type = ModelRunnerType::TT_TTS; }

  // Scheduler batching.
  size_t maxBatchSize = defaults::TTS_MAX_BATCH_SIZE;

  // Scheduler lifecycle and capacity.
  size_t maxUsers = defaults::PM_MAX_USERS;
  unsigned connectTimeoutMs = defaults::PM_CONNECT_TIMEOUT_MS;
  unsigned outputHangTimeoutMs = defaults::OUTPUT_HANG_TIMEOUT_MS;

  // Parent/worker IPC queue capacities.
  size_t taskQueueCapacity = defaults::MAX_QUEUE_SIZE;
  size_t audioQueueCapacity = defaults::TTS_AUDIO_QUEUE_CAPACITY;
  size_t cancelQueueCapacity = defaults::CANCEL_QUEUE_CAPACITY;

  // Chunk contract shared by the API, runner, and scheduler.
  uint32_t chunkTokens = defaults::TTS_CHUNK_TOKENS;

  // Tokenizer for the SpeechLM backbone, including TTS audio/speech tokens.
  std::string tokenizerPath;

  // Audio format contract for the TTS voice encoder and decoder.
  uint32_t voiceSampleRateHz = defaults::TTS_VOICE_SAMPLE_RATE_HZ;
  uint16_t voiceChannels = defaults::TTS_VOICE_CHANNELS;
  uint32_t audioSampleRateHz = defaults::TTS_AUDIO_SAMPLE_RATE_HZ;
  uint16_t audioChannels = defaults::TTS_AUDIO_CHANNELS;

  // Literal BOS token prepended to the compiled prompt; empty = none.
  // Read from the tokenizer's tokenizer_config.json in ttsEngineConfig().
  std::string bosToken;

  // Socket descriptor prefixes written by the model launcher into /dev/shm.
  std::string encoderSocketDescriptorPrefix =
      defaults::TTS_ENCODER_SOCKET_DESCRIPTOR_PREFIX;
  std::string speechlmSocketDescriptorPrefix =
      defaults::TTS_SPEECHLM_SOCKET_DESCRIPTOR_PREFIX;
  std::string decoderSocketDescriptorPrefix =
      defaults::TTS_DECODER_SOCKET_DESCRIPTOR_PREFIX;
};

using RunnerConfig =
    std::variant<BlazeConfig, EmbeddingConfig, ImageConfig, TtsConfig>;

}  // namespace tt::config
