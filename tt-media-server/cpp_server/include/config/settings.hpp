// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <string>

#include "config/runner_config.hpp"
#include "config/types.hpp"

namespace tt::config {

/**
 * Central settings: use config/defaults when env is not set; env overrides when
 * present. Same model as tt-media-server/config/settings.py. All defaults live
 * in constants.hpp defaults.
 */

/** Model service from MODEL_SERVICE. Default from defaults::MODEL_SERVICE. */
ModelService modelService();

/** True when model_service() == EMBEDDING. */
bool isEmbeddingService();

/** True when model_service() == LLM. */
bool isLlmServiceEnabled();

/** Get runner type string based on current model service configuration. */
std::string runnerType();

/** Number of worker processes = number of bracket pairs in DEVICE_IDS. */
size_t numWorkers();

/** Max wait (ms) to fill a batch. From MAX_BATCH_DELAY_TIME_MS. Default:
 * defaults::MAX_BATCH_DELAY_TIME_MS. */
unsigned batchTimeoutMs();

/** Path prepended to Python sys.path for embedding runner. From TT_PYTHON_PATH.
 * Default: defaults::TT_PYTHON_PATH. */
std::string pythonPath();

/** Tokenizer path: tokenizers/<model>/tokenizer.json relative to executable.
 * Empty if not found. No-arg overload uses the current model_type(). */
std::string tokenizerPath();
std::string tokenizerPath(ModelType model);

/** Tokenizer config path: tokenizers/<model>/tokenizer_config.json relative to
 * executable. Empty if not found. No-arg overload uses the current
 * model_type(). */
std::string tokenizerConfigPath();
std::string tokenizerConfigPath(ModelType model);

/**
 * Parse DEVICE_IDS and return the content inside the Nth bracket pair.
 * DEVICE_IDS format: "(0,1,2,3),(4,5,6,7)" → worker 0 gets "0,1,2,3", worker 1
 * gets "4,5,6,7". This value is both the worker's identity and its
 * TT_VISIBLE_DEVICES value, matching the Python scheduler flow in
 * model_services/scheduler.py.
 */
std::string visibleDevicesForWorker(size_t workerIndex);

/** Model type derived from LLM_DEVICE_BACKEND (llama -> LLAMA_3_1_8B_INSTRUCT,
 * else DEEPSEEK_R1_0528). */
ModelType modelType();

/** LLM mode from LLM_MODE. Default: defaults::LLM_MODE ("regular"). */
LLMMode llmMode();

/** Socket host from SOCKET_HOST. Default: defaults::SOCKET_HOST. */
std::string socketHost();

/** Socket port from SOCKET_PORT. Default: defaults::SOCKET_PORT. */
uint16_t socketPort();

/** Enable accumulated streaming from ENABLE_ACCUMULATED_STREAMING. Default:
 * defaults::ENABLE_ACCUMULATED_STREAMING. */
bool enableAccumulatedStreaming();

/** Max accumulated tokens from MAX_ACCUMULATED_TOKENS. Default:
 * defaults::MAX_ACCUMULATED_TOKENS. */
size_t maxAccumulatedTokens();

/** Max in-flight requests before 429. From MAX_QUEUE_SIZE. Default:
 * defaults::MAX_QUEUE_SIZE. */
size_t maxQueueSize();

/** Scheduling policy from SCHEDULING_POLICY. Default:
 * defaults::SCHEDULING_POLICY ("prefill_first"). */
SchedulingPolicy schedulingPolicy();

/** Max in-flight requests from MAX_IN_FLIGHT_COUNT. Default:
 * defaults::MAX_IN_FLIGHT_COUNT. */
size_t maxInFlightCount();

/** Max sessions count from MAX_SESSIONS_COUNT. Default:
 * defaults::MAX_SESSIONS_COUNT. */
size_t maxSessionsCount();

/** Session eviction rate percentage from SESSION_EVICTION_RATE. Default:
 * defaults::SESSION_EVICTION_RATE. */
unsigned sessionEvictionRate();

/** Number of sessions to evict at once from SESSION_EVICTION_COUNT. Default:
 * defaults::SESSION_EVICTION_COUNT. */
size_t sessionEvictionCount();

/** Max tokens to prefill on decode server from MAX_TOKENS_TO_PREFILL_ON_DECODE.
 * Default: defaults::MAX_TOKENS_TO_PREFILL_ON_DECODE. */
size_t maxTokensToPrefillOnDecode();

/** Kafka broker addresses from KAFKA_BROKERS. Default:
 * defaults::KAFKA_BROKERS. */
std::string kafkaBrokers();

/** Kafka topic name from KAFKA_OFFLOAD_TOPIC_NAME. Default:
 * defaults::KAFKA_OFFLOAD_TOPIC_NAME. */
std::string kafkaOffloadTopicName();

/** Kafka consumer group ID from KAFKA_GROUP_ID. Default:
 * defaults::KAFKA_GROUP_ID. */
std::string kafkaGroupId();

/** Max retries for session slot allocation from
 * SESSION_ALLOCATION_MAX_RETRIES. Default:
 * defaults::SESSION_ALLOCATION_MAX_RETRIES. */
unsigned sessionAllocationMaxRetries();

/** Prefill timeout in milliseconds from PREFILL_TIMEOUT_MS. Default:
 * defaults::PREFILL_TIMEOUT_MS. */
unsigned prefillTimeoutMs();

/** H2D socket ID from H2D_SOCKET_ID. Default: defaults::H2D_SOCKET_ID. */
std::string h2dSocketId();

/** D2H socket ID from D2H_SOCKET_ID. Default: defaults::D2H_SOCKET_ID. */
std::string d2hSocketId();

/** Pipeline manager connect timeout (ms) from PM_CONNECT_TIMEOUT_MS. Default:
 * defaults::PM_CONNECT_TIMEOUT_MS. */
unsigned pmConnectTimeoutMs();

/** Pipeline manager max users from PM_MAX_USERS. Default:
 * defaults::PM_MAX_USERS. */
size_t pmMaxUsers();

/** Warmup timeout (ms) while waiting for the first token during runner warmup.
 * From WARMUP_TIMEOUT_MS. Default: defaults::WARMUP_TIMEOUT_MS. */
unsigned warmupTimeoutMs();

/** Task queue name from TT_TASK_QUEUE. Default: defaults::TT_TASK_QUEUE. */
std::string ttTaskQueueName();

/** Result queue name from TT_RESULT_QUEUE. Default: defaults::TT_RESULT_QUEUE.
 */
std::string ttResultQueueName();

/** Cancel queue name from TT_CANCEL_QUEUE. Default: defaults::TT_CANCEL_QUEUE.
 */
std::string ttCancelQueueName();

/** Memory request queue name from TT_MEMORY_REQUEST_QUEUE. Default:
 * defaults::TT_MEMORY_REQUEST_QUEUE. */
std::string ttMemoryRequestQueueName();

/** Warmup signals queue name from TT_WARMUP_SIGNALS_QUEUE. Default:
 * defaults::TT_WARMUP_SIGNALS_QUEUE. */
std::string ttWarmupSignalsQueueName();

/** Memory result queue name from TT_MEMORY_RESULT_QUEUE. Default:
 * defaults::TT_MEMORY_RESULT_QUEUE. */
std::string ttMemoryResultQueueName();

/** POSIX shared-memory segment name backing the worker metrics transport.
 * From TT_WORKER_METRICS_SHM. Default: defaults::TT_WORKER_METRICS_SHM.
 * Inherited across fork+execv so main and worker resolve to the same name. */
std::string workerMetricsShmName();

/** Use DeepSeek markdown format from USE_DEEPSEEK_MD_FORMAT. Default:
 * defaults::USE_DEEPSEEK_MD_FORMAT. */
bool useDeepseekMdFormat();

/** Build LLMConfig from environment variables and runtime settings. Implemented
 * in src/config/settings.cpp. */
LLMConfig llmEngineConfig();

/** Model from MODEL. Default: defaults::MODEL. */
Model model();

}  // namespace tt::config
