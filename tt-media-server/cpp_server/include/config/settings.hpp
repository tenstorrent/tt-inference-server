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
bool isLlmService();

/** True when model_service() == IMAGE. */
bool isImageService();

/** Get runner type string based on current model service configuration. */
std::string runnerType();

/** Number of worker processes = number of bracket pairs in DEVICE_IDS. */
size_t numWorkers();

/**
 * Size of the process-wide ThreadPool that fronts inference dispatch (used by
 * `tt::utils::controllerCallbackPool()`). HTTP requests block one of these
 * threads for the full inference latency, so this caps the in-flight
 * dispatch concurrency. From `CALLBACK_POOL_THREADS`; if unset or 0,
 * auto-scales to `max(numWorkers(), CALLBACK_POOL_THREADS_MIN)` and is clamped
 * to `CALLBACK_POOL_THREADS_MAX`. Auto-scaling ensures the pool never silently
 * caps below the per-deploy `DEVICE_IDS` worker count (e.g. 32 on Galaxy).
 */
size_t callbackPoolThreads();

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

/** Model type derived from LLM_DEVICE_BACKEND:
 * "llama" -> LLAMA_3_1_8B_INSTRUCT,
 * "kimi" -> KIMI_K2_6,
 * otherwise -> DEEPSEEK_R1_0528. */
ModelType modelType();

/** LLM mode from LLM_MODE. Default: defaults::LLM_MODE ("regular"). */
LLMMode llmMode();

/** Socket host from SOCKET_HOST. Default: defaults::SOCKET_HOST. */
std::string socketHost();

/** Socket port from SOCKET_PORT. Default: defaults::SOCKET_PORT. */
uint16_t socketPort();

/** Socket transport type from SOCKET_TRANSPORT. Values: "tcp", "zmq".
 * Default: defaults::SOCKET_TRANSPORT. */
std::string socketTransport();

/** Whether the inter-server socket integrates with PrefillGateway. From
 * USE_PREFILL_GATEWAY. */
bool usePrefillGateway();

/** Prefill identity for PrefillRegistrationMessage; falls back to
 * "<hostname>:<SOCKET_PORT>". From PREFILL_SERVER_ID. */
std::string prefillServerId();

/**
 * Role shown in every log line, e.g. "decode", "prefill", "prefill-worker0".
 * The role is the LLM_MODE (decode/prefill/regular) for the LLM service or the
 * service name (image/embedding) otherwise; a forked worker subprocess appends
 * "-worker<index>" so worker lines are distinguishable from the HTTP node.
 *
 * @param workerIndex >=0 for a forked worker subprocess; appends
 *   "-worker<index>" to the role (e.g. "decode-worker0").
 */
std::string logInstanceTag(int workerIndex = -1);

/** Capacity hint for the gateway, 0 = unlimited. From PREFILL_MAX_IN_FLIGHT. */
uint32_t prefillMaxInFlight();

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
 * defaults::MAX_SESSIONS_COUNT. Can be overridden at runtime via
 * setMaxSessionsCount(). */
size_t maxSessionsCount();

/** Set max sessions count override. Pass 0 to clear the override and use the
 * environment variable value. */
void setMaxSessionsCount(size_t count);

/** Session eviction rate percentage from SESSION_EVICTION_RATE. Default:
 * defaults::SESSION_EVICTION_RATE. */
unsigned sessionEvictionRate();

/** Number of sessions to evict at once from SESSION_EVICTION_COUNT. Default:
 * defaults::SESSION_EVICTION_COUNT. */
size_t sessionEvictionCount();

/** Max tokens to prefill on decode server from MAX_TOKENS_TO_PREFILL_ON_DECODE.
 * Default: defaults::MAX_TOKENS_TO_PREFILL_ON_DECODE. */
size_t maxTokensToPrefillOnDecode();

/** Max context length (prompt + completion) from MAX_CONTEXT_LENGTH. Default:
 * defaults::MAX_CONTEXT_LENGTH. */
size_t maxContextLength();

/** Max input sequence length (prompt tokens) from MAX_ISL. Default:
 * defaults::MAX_ISL. */
size_t maxISL();

/** Minimum matched tokens required to justify a slot copy operation.
 * From MIN_TOKENS_TO_COPY. Default: defaults::MIN_TOKENS_TO_COPY. */
size_t minTokensToCopy();

/** KV cache block size from KV_CACHE_BLOCK_SIZE. Default:
 * defaults::KV_CACHE_BLOCK_SIZE. */
size_t kvCacheBlockSize();

/** KV cache first block size from KV_CACHE_FIRST_BLOCK_SIZE. Default:
 * defaults::KV_CACHE_FIRST_BLOCK_SIZE. */
size_t kvCacheFirstBlockSize();

/** Minimum match percentage for prefix cache hit from
 * PREFIX_CACHE_HIT_THRESHOLD. Default: defaults::PREFIX_CACHE_HIT_THRESHOLD.
 * Set to 0 to disable threshold check (accept any match). */
float prefixCacheHitThreshold();

/** Use fast mode from USE_FAST_MODE. Default: defaults::USE_FAST_MODE. */
bool useFastMode();

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

/** Blaze socket descriptor prefix from BLAZE_SOCKET_DESCRIPTOR_PREFIX. Default:
 * defaults::BLAZE_SOCKET_DESCRIPTOR_PREFIX. */
std::string blazeSocketDescriptorPrefix();

/** Pipeline manager connect timeout (ms) from PM_CONNECT_TIMEOUT_MS. Default:
 * defaults::PM_CONNECT_TIMEOUT_MS. */
unsigned pmConnectTimeoutMs();

/** Decode scheduler max users from DS_MAX_USERS. Default:
 * defaults::DS_MAX_USERS. */
size_t dsMaxUsers();

/** Prefill scheduler max users from PS_MAX_USERS. Default:
 * defaults::PS_MAX_USERS. */
size_t psMaxUsers();

/** Prefill H2D service ID from PREFILL_H2D_SERVICE_ID. Default:
 * defaults::PREFILL_H2D_SERVICE_ID. */
std::string prefillH2DServiceId();

/** Prefill number of layers from PREFILL_NUM_LAYERS. Default:
 * defaults::PREFILL_NUM_LAYERS. */
std::string prefillNumLayers();
/** Warmup timeout (ms) while waiting for the first token during runner warmup.
 * From WARMUP_TIMEOUT_MS. Default: defaults::WARMUP_TIMEOUT_MS. */
unsigned warmupTimeoutMs();

/** Max time (ms) without any model output while at least one request is in
 * flight before the runner self-terminates the worker process. From
 * OUTPUT_HANG_TIMEOUT_MS. Default: defaults::OUTPUT_HANG_TIMEOUT_MS. */
unsigned outputHangTimeoutMs();

/** Task queue name from TT_TASK_QUEUE. Default: defaults::TT_TASK_QUEUE. */
std::string ttTaskQueueName();

/** Result queue name from TT_RESULT_QUEUE. Default: defaults::TT_RESULT_QUEUE.
 */
std::string ttResultQueueName();

/** Cancel queue name from TT_CANCEL_QUEUE. Default: defaults::TT_CANCEL_QUEUE.
 */
std::string ttCancelQueueName();

/** Media payload task queue name from TT_MEDIA_TASK_QUEUE. */
std::string ttMediaTaskQueueName();

/** Media payload result queue name from TT_MEDIA_RESULT_QUEUE. */
std::string ttMediaResultQueueName();

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

std::string wireFormat();

// IPC queue capacities - configurable via environment variables
/** Result queue capacity from RESULT_QUEUE_CAPACITY. Default:
 * defaults::RESULT_QUEUE_CAPACITY. */
size_t resultQueueCapacity();

/** Cancel queue capacity from CANCEL_QUEUE_CAPACITY. Default:
 * defaults::CANCEL_QUEUE_CAPACITY. */
size_t cancelQueueCapacity();

/** Memory queue capacity from MEMORY_QUEUE_CAPACITY. Default:
 * defaults::MEMORY_QUEUE_CAPACITY. */
size_t memoryQueueCapacity();

// Shared memory slot buffer constants
/** SHM slots from SHM_SLOTS. Default: defaults::SHM_SLOTS. */
int shmSlots();

/** Prefill max token IDs from PREFILL_MAX_TOKEN_IDS. Default:
 * defaults::PREFILL_MAX_TOKEN_IDS. */
int prefillMaxTokenIds();

/** Decode max token IDs from DECODE_MAX_TOKEN_IDS. Default:
 * defaults::DECODE_MAX_TOKEN_IDS. */
int decodeMaxTokenIds();

// ---------------------------------------------------------------------------
// Dynamo TCP backend (NVIDIA Dynamo frontend integration)
// ---------------------------------------------------------------------------

/** Whether the Dynamo TCP `generate` endpoint should bind on startup. From
 * DYNAMO_ENDPOINT_ENABLED. Default: defaults::DYNAMO_ENDPOINT_ENABLED. */
bool dynamoEndpointEnabled();

/** Bind host for the Dynamo listener. From DYNAMO_BIND_HOST. Default:
 * defaults::DYNAMO_BIND_HOST. */
std::string dynamoBindHost();

/** Etcd endpoint(s) the discovery client dials. From DYNAMO_ETCD_ENDPOINTS,
 * falling back to ETCD_ENDPOINTS (the env var Dynamo's own runtime reads).
 * Default: defaults::DYNAMO_ETCD_ENDPOINTS. */
std::string dynamoEtcdEndpoints();

/** Lease TTL (seconds) for etcd-backed discovery. From
 * DYNAMO_ETCD_LEASE_TTL_SECS. Default: defaults::DYNAMO_ETCD_LEASE_TTL_SECS. */
int64_t dynamoEtcdLeaseTtlSecs();

/** Discovery namespace key. From DYNAMO_NAMESPACE. Default:
 * defaults::DYNAMO_NAMESPACE. */
std::string dynamoNamespace();

/** Discovery component key. From DYNAMO_COMPONENT. Default:
 * defaults::DYNAMO_COMPONENT. */
std::string dynamoComponent();

/** Discovery endpoint key. From DYNAMO_ENDPOINT_NAME. Default:
 * defaults::DYNAMO_ENDPOINT_NAME. */
std::string dynamoEndpointName();

/** Build LLMConfig from environment variables and runtime settings. Implemented
 * in src/config/settings.cpp. */
LLMConfig llmEngineConfig();

/** Build ImageConfig from environment variables and runtime settings. Reads
 * MODEL_RUNNER_TYPE, MAX_BATCH_SIZE, SDXL_IMAGE_RESOLUTION. Implemented in
 * src/config/settings.cpp. */
ImageConfig imageEngineConfig();

/** Build the runner config used by a fork/exec worker for the active service.
 * Media configs receive the worker's DEVICE_IDS group as visible_devices. */
RunnerConfig workerRunnerConfig(size_t workerIndex);

/** Model from MODEL. Default: defaults::MODEL. */
Model model();

}  // namespace tt::config
