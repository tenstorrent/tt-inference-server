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

// PrefillGateway integration. When true, the inter-server socket flips its
// role to talk to the gateway instead of the peer directly:
//   - decode mode: connect (CLIENT) to gateway at SOCKET_HOST:SOCKET_PORT.
//   - prefill mode: listen (SERVER) on SOCKET_PORT for the gateway to dial in.
constexpr bool USE_PREFILL_GATEWAY = false;
// Stable identity advertised to the gateway in PrefillRegistrationMessage.
// Empty falls back to "<hostname>:<port>". Used only when prefill mode is
// running behind the gateway.
constexpr const char* PREFILL_SERVER_ID = "";
// Capacity hint the prefill advertises to the gateway. 0 = unlimited.
constexpr uint32_t PREFILL_MAX_IN_FLIGHT = 0;

constexpr size_t MAX_QUEUE_SIZE = 1000;
constexpr const char* SCHEDULING_POLICY =
    "prefill_first";  // "prefill_first" or "max_occupancy"
constexpr const char* LLM_DEVICE_BACKEND =
    "mock_pipeline";  // "mock", "mock_pipeline", "pipeline", "llama"
constexpr const bool ENABLE_ACCUMULATED_STREAMING = false;
constexpr size_t MAX_ACCUMULATED_TOKENS = 5;
constexpr size_t MAX_IN_FLIGHT_COUNT = 32;
constexpr size_t MAX_SESSIONS_COUNT = 128;
constexpr unsigned SESSION_EVICTION_RATE = 90;
constexpr size_t SESSION_EVICTION_COUNT = 10;
constexpr size_t MAX_TOKENS_TO_PREFILL_ON_DECODE = 1000;
constexpr size_t MAX_CONTEXT_LENGTH = 65536;  // 64k
constexpr bool USE_FAST_MODE = false;
constexpr const char* KAFKA_BROKERS = "localhost:9092";
constexpr const char* KAFKA_OFFLOAD_TOPIC_NAME = "session-offload";
constexpr const char* KAFKA_GROUP_ID = "migration-workers";

constexpr unsigned SESSION_ALLOCATION_MAX_RETRIES = 15;
constexpr unsigned PREFILL_TIMEOUT_MS = 20000;

constexpr const char* BLAZE_SOCKET_DESCRIPTOR_PREFIX = "deepseek";
constexpr const char* TT_TASK_QUEUE = "tt_tasks";
constexpr const char* TT_RESULT_QUEUE = "tt_results";
constexpr const char* TT_CANCEL_QUEUE = "tt_cancels";
constexpr const char* TT_WARMUP_SIGNALS_QUEUE = "tt_warmup_signals";
constexpr const char* TT_MEMORY_REQUEST_QUEUE = "tt_mem_requests";
constexpr const char* TT_MEMORY_RESULT_QUEUE = "tt_mem_results";
constexpr const char* TT_WORKER_METRICS_SHM = "/tt_worker_metrics";
constexpr unsigned PM_CONNECT_TIMEOUT_MS = 30000;
constexpr size_t PM_MAX_USERS = 128;
constexpr bool USE_DEEPSEEK_MD_FORMAT = false;
constexpr unsigned WARMUP_TIMEOUT_MS = 10000;
/**
 * Max time (ms) the runner may go without producing a model output while at
 * least one request is in flight before it self-terminates the worker
 * process. Self-terminating lets the infrastructure monitoring stack notice
 * the crash and restart the server instead of hanging silently.
 */
constexpr unsigned OUTPUT_HANG_TIMEOUT_MS = 60000;

constexpr const char* MODEL = "deepseek-ai/DeepSeek-R1-0528";

constexpr const char* SERVER_HOST = "0.0.0.0";
constexpr uint16_t SERVER_PORT = 8000;
constexpr size_t MAX_CONNECTIONS = 100000;
constexpr size_t IDLE_CONNECTION_TIMEOUT_S = 3600;
constexpr size_t CLIENT_MAX_BODY_BYTES = 100 * 1024 * 1024;  // 100 MB
constexpr size_t LOG_FILE_MAX_BYTES = 50 * 1024 * 1024;      // 50 MB
constexpr size_t LOG_FILE_MAX_COUNT = 5;
constexpr size_t EMBEDDING_MAX_PIPE_BYTES = 100 * 1024 * 1024;  // 100 MB
constexpr int CALLBACK_POOL_THREADS = 16;
constexpr unsigned WORKER_STOP_TIMEOUT_MS = 500;
constexpr unsigned SHUTDOWN_POLL_MS = 50;

// IPC queue capacities
constexpr size_t RESULT_QUEUE_CAPACITY = 65536;
constexpr size_t CANCEL_QUEUE_CAPACITY = 1024;
constexpr size_t MEMORY_QUEUE_CAPACITY = 128;

// IPC message sizes
constexpr size_t MAX_SEQUENCE_NON_TOKEN_BYTES = 4096;
constexpr size_t TASK_QUEUE_MAX_MSG_SIZE =
    MAX_CONTEXT_LENGTH * sizeof(int64_t) + MAX_SEQUENCE_NON_TOKEN_BYTES;
constexpr size_t MEMORY_REQUEST_MAX_MSG_SIZE = 256;
constexpr size_t MEMORY_RESULT_MAX_MSG_SIZE = 4096;

// Shared memory slot buffer constants
constexpr int SHM_SLOTS = 64;
constexpr int PREFILL_MAX_TOKEN_IDS = 131072;  // upper bound for prefill prompt
constexpr int DECODE_MAX_TOKEN_IDS = 1;

}  // namespace tt::config::defaults
