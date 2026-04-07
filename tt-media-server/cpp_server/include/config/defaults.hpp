// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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
constexpr size_t MAX_QUEUE_SIZE = 1000;
constexpr const char* SCHEDULING_POLICY =
    "prefill_first";  // "prefill_first" or "max_occupancy"
constexpr const char* LLM_DEVICE_BACKEND =
    "mock_pipeline";  // "mock", "mock_pipeline", "pipeline", "llama"
constexpr const bool ENABLE_ACCUMULATED_STREAMING = false;
constexpr size_t MAX_ACCUMULATED_TOKENS = 5;
constexpr size_t MAX_IN_FLIGHT_COUNT = 32;
constexpr size_t MAX_SESSIONS_COUNT = 64;
constexpr unsigned SESSION_EVICTION_RATE = 90;
constexpr size_t SESSION_EVICTION_COUNT = 10;
constexpr size_t MAX_TOKENS_TO_PREFILL_ON_DECODE = 100;

constexpr const char* SERVER_HOST = "0.0.0.0";
constexpr uint16_t SERVER_PORT = 8000;
constexpr size_t MAX_CONNECTIONS = 100000;
constexpr size_t IDLE_CONNECTION_TIMEOUT_S = 300;
constexpr size_t CLIENT_MAX_BODY_BYTES = 100 * 1024 * 1024;  // 100 MB
constexpr size_t LOG_FILE_MAX_BYTES = 50 * 1024 * 1024;      // 50 MB
constexpr size_t LOG_FILE_MAX_COUNT = 5;
constexpr size_t EMBEDDING_MAX_PIPE_BYTES = 100 * 1024 * 1024;  // 100 MB
constexpr int CALLBACK_POOL_THREADS = 16;
constexpr unsigned WORKER_STOP_TIMEOUT_MS = 500;
constexpr unsigned SHUTDOWN_POLL_MS = 50;

}  // namespace tt::config::defaults
