// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::config::defaults {

/**
 * Default values when the corresponding environment variable is not set or
 * empty. Environment variables override these when present.
 */

constexpr const char* DEVICE_IDS = "(0)";
constexpr const char* MODEL_SERVICE = "llm";
constexpr unsigned MAX_BATCH_DELAY_TIME_MS = 5;
constexpr const char* TT_PYTHON_PATH = "..";
constexpr const char* LLM_MODE = "regular";  // "regular", "prefill", "decode"
constexpr const char* SOCKET_HOST = "localhost";
constexpr uint16_t SOCKET_PORT = 9000;

// PrefillGateway integration. When true, decode connects as CLIENT to the
// gateway and prefill listens as SERVER for the gateway to dial in.
constexpr bool USE_PREFILL_GATEWAY = false;
// Stable identity sent in PrefillRegistrationMessage; empty -> "<host>:<port>".
constexpr const char* PREFILL_SERVER_ID = "";
// Capacity hint sent to the gateway. 0 = unlimited.
constexpr uint32_t PREFILL_MAX_IN_FLIGHT = 0;

constexpr size_t MAX_QUEUE_SIZE = 1000;
constexpr const char* LLM_DEVICE_BACKEND =
    "mock_scheduler";  // "mock_pipeline", "mock_scheduler", or
                       // "pipeline_manager"
constexpr size_t MAX_IN_FLIGHT_COUNT = 32;
constexpr size_t MAX_SESSIONS_COUNT = 128;
constexpr unsigned SESSION_EVICTION_RATE = 90;
constexpr size_t SESSION_EVICTION_COUNT = 10;
constexpr size_t MAX_TOKENS_TO_PREFILL_ON_DECODE = 1000;
constexpr size_t MAX_CONTEXT_LENGTH = 65536;  // 64k
constexpr size_t MAX_ISL = 256000;  // 2000k (max input sequence length)
constexpr size_t MIN_TOKENS_TO_COPY =
    1024;  // min matched tokens to justify slot copy
constexpr size_t PREFIX_CACHE_BLOCK_SIZE = 32;
constexpr size_t PREFIX_CACHE_FIRST_BLOCK_SIZE = 128;
constexpr unsigned PREFIX_CACHE_HIT_THRESHOLD = 40;
constexpr bool USE_FAST_MODE = false;
constexpr bool ENABLE_MIGRATION = false;
// PrefillScheduler drives cross-endpoint (P->D) KV migration via the
// Kafka-backed RemoteKVManagerAdapter
constexpr bool PREFILL_USE_REMOTE_KV_MANAGER = false;
constexpr const char* MIGRATION_CMD_QUEUE_NAME = "mig_ep0_cmd";
constexpr const char* MIGRATION_TABLE_QUEUE_NAME = "mig_ep0_table";
constexpr const char* MIGRATION_RESP_QUEUE_NAME = "mig_ep0_resp";
constexpr uint32_t MIGRATION_PREFILL_ENDPOINT_ID = 0;
constexpr uint32_t MIGRATION_DECODE_ENDPOINT_ID = 1;
constexpr const char* PREFILL_ACK_CHANNEL_NAME = "tt_prefill_layer_acks";
constexpr const char* KAFKA_BROKERS = "localhost:9092";
constexpr const char* KAFKA_OFFLOAD_TOPIC_NAME = "session-offload";
constexpr const char* KAFKA_GROUP_ID = "migration-workers";
constexpr const char* KAFKA_MIGRATION_REQUEST_TOPIC = "kv-migration-requests";
constexpr const char* KAFKA_MIGRATION_ACK_TOPIC = "kv-migration-acks";

// Mooncake KV Migration configuration.
constexpr unsigned KV_MIGRATION_TIMEOUT_MS = 60000;
constexpr unsigned KV_MIGRATION_SWEEP_INTERVAL_MS = 5000;
constexpr unsigned KV_MIGRATION_DRAIN_POLL_MS = 100;

constexpr unsigned SESSION_ALLOCATION_MAX_RETRIES = 15;

constexpr const char* SPEC_DECODE_MODE = "none";
constexpr size_t MTP_LEVEL = 1;

// Number of pipeline stages of the Blaze Decode model.
constexpr uint32_t BLAZE_NUMBER_OF_PIPELINE_STAGES = 64;

constexpr const char* TT_TASK_QUEUE = "tt_tasks";
constexpr const char* TT_RESULT_QUEUE = "tt_results";
constexpr const char* TT_CANCEL_QUEUE = "tt_cancels";
constexpr const char* TT_MEDIA_TASK_QUEUE = "tt_media_tasks";
constexpr const char* TT_MEDIA_RESULT_QUEUE = "tt_media_results";
constexpr const char* TT_WARMUP_SIGNALS_QUEUE = "tt_warmup_signals";
constexpr const char* TT_MEMORY_REQUEST_QUEUE = "tt_mem_requests";
constexpr const char* TT_MEMORY_RESULT_QUEUE = "tt_mem_results";
constexpr const char* TT_WORKER_METRICS_SHM = "/tt_worker_metrics";
constexpr uint32_t MODEL_NUM_LAYERS = 61;
constexpr uint32_t PREFILL_CHUNK_SIZE = 5120;
constexpr unsigned PM_CONNECT_TIMEOUT_MS = 30000;
constexpr size_t PM_MAX_USERS = 128;
constexpr unsigned WARMUP_TIMEOUT_MS = 150000;
/**
 * Max time (ms) the runner may go without producing a model output while at
 * least one request is in flight before it self-terminates the worker
 * process. Self-terminating lets the infrastructure monitoring stack notice
 * the crash and restart the server instead of hanging silently.
 */
constexpr unsigned OUTPUT_HANG_TIMEOUT_MS = 150000;

constexpr const char* MODEL = "deepseek-ai/DeepSeek-R1-0528";
constexpr const char* WIRE_FORMAT = "blaze";

constexpr const char* SERVER_HOST = "0.0.0.0";
constexpr uint16_t SERVER_PORT = 8000;
constexpr size_t MAX_CONNECTIONS = 100000;
constexpr size_t IDLE_CONNECTION_TIMEOUT_S = 3600;
constexpr size_t CLIENT_MAX_BODY_BYTES = 100 * 1024 * 1024;  // 100 MB
constexpr size_t LOG_FILE_MAX_BYTES = 50 * 1024 * 1024;      // 50 MB
constexpr size_t LOG_FILE_MAX_COUNT = 5;
constexpr size_t EMBEDDING_MAX_PIPE_BYTES = 100 * 1024 * 1024;  // 100 MB
// Lower bound used when CALLBACK_POOL_THREADS env is unset or 0; preserves
// the legacy default (16) for small (1-16 worker) deployments.
constexpr size_t CALLBACK_POOL_THREADS_MIN = 16;
// Safety ceiling for the dispatch thread pool to avoid pathological configs
// (e.g. accidental CALLBACK_POOL_THREADS=100000).
constexpr size_t CALLBACK_POOL_THREADS_MAX = 32;
constexpr unsigned WORKER_STOP_TIMEOUT_MS = 500;
constexpr unsigned SHUTDOWN_POLL_MS = 50;

// IPC queue capacities
constexpr size_t RESULT_QUEUE_CAPACITY = 65536;
constexpr size_t CANCEL_QUEUE_CAPACITY = 1024;
constexpr size_t MEMORY_QUEUE_CAPACITY = 128;

// IPC message sizes
constexpr size_t MAX_SEQUENCE_NON_TOKEN_BYTES = 4096;
constexpr size_t TASK_QUEUE_MAX_MSG_SIZE =
    MAX_CONTEXT_LENGTH * sizeof(uint32_t) + MAX_SEQUENCE_NON_TOKEN_BYTES;
constexpr size_t MEMORY_REQUEST_MAX_MSG_SIZE = 256;
constexpr size_t MEMORY_RESULT_MAX_MSG_SIZE = 4096;

// Shared memory slot buffer constants
constexpr int SHM_SLOTS = 64;
constexpr int PREFILL_MAX_TOKEN_IDS = 131072;  // upper bound for prefill prompt
constexpr int DECODE_MAX_TOKEN_IDS = 1;

// Dynamo backend (TCP `generate` endpoint that registers with NVIDIA Dynamo
// frontends). All defaults are overridable via env vars; the endpoint is
// off unless DYNAMO_ENDPOINT_ENABLED=1.
constexpr bool DYNAMO_ENDPOINT_ENABLED = false;
// When true, Dynamo owns prefill/decode routing and prefill-first
// disaggregation is enabled (etcd discovers decode; ZMQ reserves slots).
constexpr bool DYNAMO_ROUTING = false;
constexpr const char* DYNAMO_BIND_HOST = "0.0.0.0";
constexpr uint16_t DYNAMO_BIND_PORT = 0;  // 0 = OS-assigned ephemeral port.
constexpr const char* DYNAMO_NAMESPACE = "default";
constexpr const char* DYNAMO_COMPONENT = "backend";
constexpr const char* DYNAMO_ENDPOINT_NAME = "generate";

// Discovery: etcd endpoint for Dynamo's KVStoreDiscovery.
constexpr const char* DYNAMO_ETCD_ENDPOINTS = "http://etcd:2379/";
// Lease TTL for instance + MDC entries in etcd. The keep-alive thread
// refreshes the lease at half this interval so a missed tick doesn't trip
// the reaper.
constexpr int64_t DYNAMO_ETCD_LEASE_TTL_SECS = 10;

constexpr unsigned MOCK_PREFILL_CHUNK_LATENCY_MS = 1353;
constexpr unsigned MOCK_STAGE_LATENCY_US = 44;
constexpr uint32_t MOCK_PIPELINE_STAGES = 64;
constexpr uint32_t MOCK_PREFILL_CHUNK_SIZE = 24;
constexpr unsigned MOCK_DECODE_TOKEN_ID = 12345;

}  // namespace tt::config::defaults
